package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// chunk writes one chat.completion.chunk as an SSE data line.
func chunk(w http.ResponseWriter, data string) {
	fmt.Fprintf(w, "data: %s\n\n", data)
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

func TestChatCompletion_OpenAIStream_Text(t *testing.T) {
	var gotStream, gotIncludeUsage bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		gotStream, _ = body["stream"].(bool)
		if so, ok := body["stream_options"].(map[string]any); ok {
			gotIncludeUsage, _ = so["include_usage"].(bool)
		}

		w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		w.WriteHeader(http.StatusOK)
		// vLLM shape (what the Maple proxy emits): role-only first delta, usage
		// on every chunk, reasoning_content interleaved, a trailing chunk with
		// empty choices, then [DONE].
		chunk(w, `{"id":"c1","model":"kimi-k3","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}],"usage":{"prompt_tokens":42,"completion_tokens":0}}`)
		chunk(w, `{"id":"c1","model":"kimi-k3","choices":[{"index":0,"delta":{"reasoning_content":"let me "},"finish_reason":null}],"usage":{"prompt_tokens":42,"completion_tokens":2}}`)
		chunk(w, `{"id":"c1","model":"kimi-k3","choices":[{"index":0,"delta":{"reasoning":"think"},"finish_reason":null}],"usage":{"prompt_tokens":42,"completion_tokens":3}}`)
		chunk(w, `{"id":"c1","model":"kimi-k3","choices":[{"index":0,"delta":{"content":"{\"a\":"},"finish_reason":null}],"usage":{"prompt_tokens":42,"completion_tokens":5}}`)
		chunk(w, `{"id":"c1","model":"kimi-k3","choices":[{"index":0,"delta":{"content":"1}"},"finish_reason":"stop"}],"usage":{"prompt_tokens":42,"completion_tokens":7}}`)
		chunk(w, `{"id":"c1","model":"kimi-k3","choices":[],"usage":{"prompt_tokens":42,"completion_tokens":7}}`)
		chunk(w, `[DONE]`)
	}))
	defer srv.Close()

	c := NewClient(ClientConfig{BaseURL: srv.URL, Provider: "openai", Token: "k", Timeout: 5 * time.Second})
	resp, err := c.ChatCompletion(context.Background(), ChatRequest{
		Model:     "kimi-k3",
		Messages:  []Message{{Role: "user", Content: "hi"}},
		MaxTokens: 64,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !gotStream || !gotIncludeUsage {
		t.Fatalf("request did not ask to stream with usage (stream=%v include_usage=%v)", gotStream, gotIncludeUsage)
	}
	if resp.Content != `{"a":1}` {
		t.Fatalf("Content = %q, want reassembled deltas", resp.Content)
	}
	if resp.FinishReason != "stop" {
		t.Fatalf("FinishReason = %q, want stop", resp.FinishReason)
	}
	if resp.Usage.PromptTokens != 42 || resp.Usage.CompletionTokens != 7 {
		t.Fatalf("Usage = %+v, want 42/7 from the trailing chunk", resp.Usage)
	}
	if resp.Usage.ThinkingChars != len("let me think") {
		t.Fatalf("ThinkingChars = %d, reasoning deltas must be counted, not kept", resp.Usage.ThinkingChars)
	}
	if resp.Model != "kimi-k3" {
		t.Fatalf("Model = %q", resp.Model)
	}
}

func TestChatCompletion_OpenAIStream_OpenAIShapeUsageOnlyAtEnd(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		// OpenAI shape: no usage until the final empty-choices chunk, finish_reason
		// on the last content chunk, content split mid-token.
		chunk(w, `{"model":"gpt-5.6","choices":[{"index":0,"delta":{"content":"hel"},"finish_reason":null}]}`)
		chunk(w, `{"model":"gpt-5.6","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":"length"}]}`)
		chunk(w, `{"model":"gpt-5.6","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2}}`)
		chunk(w, `[DONE]`)
	}))
	defer srv.Close()

	c := NewClient(ClientConfig{BaseURL: srv.URL, Provider: "openai", Token: "k", Timeout: 5 * time.Second})
	resp, err := c.ChatCompletion(context.Background(), ChatRequest{Model: "gpt-5.6", Messages: []Message{{Role: "user", Content: "x"}}, MaxTokens: 2})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != "hello" || resp.FinishReason != "length" {
		t.Fatalf("got %q / %q", resp.Content, resp.FinishReason)
	}
	if resp.Usage.CompletionTokens != 2 {
		t.Fatalf("usage not taken from trailing chunk: %+v", resp.Usage)
	}
}

func TestChatCompletion_OpenAIStream_ToolCallArguments(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		chunk(w, `{"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"security_analysis","arguments":"{\"find"}}]},"finish_reason":null}]}`)
		chunk(w, `{"model":"m","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ings\":[1,2]}"}}]},"finish_reason":null}]}`)
		chunk(w, `{"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`)
		chunk(w, `[DONE]`)
	}))
	defer srv.Close()

	c := NewClient(ClientConfig{BaseURL: srv.URL, Provider: "openai", Token: "k", Timeout: 5 * time.Second})
	schema := json.RawMessage(`{"type":"object"}`)
	resp, err := c.ChatCompletion(context.Background(), ChatRequest{
		Model: "m", Messages: []Message{{Role: "user", Content: "go"}}, MaxTokens: 64,
		OutputMode: OutputModeToolUse, ResponseSchema: &schema,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if resp.Content != `{"findings":[1,2]}` {
		t.Fatalf("Content = %q, want reassembled tool arguments", resp.Content)
	}
}

func TestChatCompletion_OpenAIStream_Truncated(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		chunk(w, `{"model":"m","choices":[{"index":0,"delta":{"content":"{\"partial\":"},"finish_reason":null}]}`)
		// connection ends here: no finish_reason, no [DONE]
	}))
	defer srv.Close()

	c := NewClient(ClientConfig{BaseURL: srv.URL, Provider: "openai", Token: "k", Timeout: 5 * time.Second, MaxRetries: 0})
	_, err := c.ChatCompletion(context.Background(), ChatRequest{Model: "m", Messages: []Message{{Role: "user", Content: "x"}}, MaxTokens: 64})
	if err == nil {
		t.Fatal("expected error on truncated stream, got nil")
	}
	if !strings.Contains(err.Error(), "[DONE]") {
		t.Fatalf("error = %q, want mention of [DONE]", err)
	}
}

func TestChatCompletion_OpenAIStream_MidStreamError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		chunk(w, `{"model":"m","choices":[{"index":0,"delta":{"content":"a"},"finish_reason":null}]}`)
		chunk(w, `{"error":{"message":"upstream overloaded","type":"server_error"}}`)
	}))
	defer srv.Close()

	c := NewClient(ClientConfig{BaseURL: srv.URL, Provider: "openai", Token: "k", Timeout: 5 * time.Second, MaxRetries: 0})
	_, err := c.ChatCompletion(context.Background(), ChatRequest{Model: "m", Messages: []Message{{Role: "user", Content: "x"}}, MaxTokens: 64})
	if err == nil || !strings.Contains(err.Error(), "upstream overloaded") {
		t.Fatalf("error = %v, want the mid-stream error surfaced", err)
	}
}

func TestChatCompletion_OpenAIBlockingStillParsed(t *testing.T) {
	// A server that ignores stream:true and answers with JSON must still work.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, successResponse("plain", "m", "stop", 3, 1))
	}))
	defer srv.Close()

	c := NewClient(ClientConfig{BaseURL: srv.URL, Provider: "openai", Token: "k", Timeout: 5 * time.Second})
	resp, err := c.ChatCompletion(context.Background(), ChatRequest{Model: "m", Messages: []Message{{Role: "user", Content: "x"}}, MaxTokens: 8})
	if err != nil || resp.Content != "plain" {
		t.Fatalf("blocking fallback broken: %v %+v", err, resp)
	}
}

func TestBuildRequestBody_DatabricksDoesNotStream(t *testing.T) {
	c := &httpClient{provider: "databricks"}
	body, err := c.buildRequestBody(ChatRequest{Model: "m", Messages: []Message{{Role: "user", Content: "x"}}, MaxTokens: 1}, false, true, false)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(body), `"stream"`) {
		t.Fatalf("databricks request must not stream: %s", body)
	}
}
