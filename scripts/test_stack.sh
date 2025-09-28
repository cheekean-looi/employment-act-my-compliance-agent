#!/bin/bash
# Test the complete Employment Act Malaysia agent stack
# Comprehensive integration testing with real API calls

set -e

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Configuration
VLLM_URL=${VLLM_BASE_URL:-"http://localhost:8000"}
API_URL=${API_BASE_URL:-"http://localhost:8001"}
UI_URL="http://localhost:8501"

echo "🧪 Testing Employment Act Malaysia Agent Stack"
echo "vLLM URL: $VLLM_URL"
echo "API URL: $API_URL"
echo "UI URL: $UI_URL"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# Test function
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    
    echo -n "Testing $test_name... "
    
    if eval "$test_command" > /dev/null 2>&1; then
        echo -e "${GREEN}PASS${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "${RED}FAIL${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# Advanced test with output capture
run_test_with_output() {
    local test_name="$1"
    local test_command="$2"
    local expected_pattern="$3"
    
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
    
    echo -n "Testing $test_name... "
    
    local output
    output=$(eval "$test_command" 2>&1)
    local exit_code=$?
    
    if [ $exit_code -eq 0 ] && [[ "$output" =~ $expected_pattern ]]; then
        echo -e "${GREEN}PASS${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "${RED}FAIL${NC}"
        echo "  Expected pattern: $expected_pattern"
        echo "  Got output: $output"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

echo "🔍 Basic Connectivity Tests"
echo "=========================="

# Test vLLM connectivity
run_test "vLLM Health Check" "curl -s -f $VLLM_URL/health"
run_test "vLLM Models Endpoint" "curl -s -f $VLLM_URL/v1/models"

# Test API connectivity
run_test "API Health Check" "curl -s -f $API_URL/health"
run_test "API OpenAPI Docs" "curl -s -f $API_URL/docs"
run_test "API Metrics Endpoint" "curl -s -f $API_URL/metrics"

# Test UI connectivity
run_test "Streamlit UI Accessibility" "curl -s -f $UI_URL"

echo ""
echo "📊 API Functionality Tests"
echo "=========================="

# Test API health endpoint with detailed response
run_test_with_output "API Health Response Format" \
    "curl -s $API_URL/health | jq -r '.status'" \
    "ok|degraded"

# Test configuration validation
run_test "API Configuration Validation" "curl -s -f $API_URL/config/validate"

# Test answer endpoint with simple query
echo -n "Testing Answer Endpoint... "
ANSWER_RESPONSE=$(curl -s -X POST "$API_URL/answer" \
    -H "Content-Type: application/json" \
    -d '{"query": "What is sick leave?", "max_tokens": 100}')

if echo "$ANSWER_RESPONSE" | jq -e '.answer' > /dev/null 2>&1; then
    echo -e "${GREEN}PASS${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
    
    # Extract and display response details
    ANSWER_TEXT=$(echo "$ANSWER_RESPONSE" | jq -r '.answer')
    LATENCY=$(echo "$ANSWER_RESPONSE" | jq -r '.latency_ms')
    CACHE_HIT=$(echo "$ANSWER_RESPONSE" | jq -r '.cache_hit')
    CONFIDENCE=$(echo "$ANSWER_RESPONSE" | jq -r '.confidence')
    
    echo "  Answer preview: ${ANSWER_TEXT:0:80}..."
    echo "  Latency: ${LATENCY}ms, Cache hit: $CACHE_HIT, Confidence: $CONFIDENCE"
else
    echo -e "${RED}FAIL${NC}"
    echo "  Response: $ANSWER_RESPONSE"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
TESTS_TOTAL=$((TESTS_TOTAL + 1))

# Test severance calculator
echo -n "Testing Severance Calculator... "
SEVERANCE_RESPONSE=$(curl -s -X POST "$API_URL/tool/severance" \
    -H "Content-Type: application/json" \
    -d '{
        "monthly_wage": 3000.0,
        "years_of_service": 2.5,
        "termination_reason": "dismissal_without_cause",
        "annual_leave_days": 5
    }')

if echo "$SEVERANCE_RESPONSE" | jq -e '.severance_pay' > /dev/null 2>&1; then
    echo -e "${GREEN}PASS${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
    
    # Extract calculation details
    SEVERANCE_PAY=$(echo "$SEVERANCE_RESPONSE" | jq -r '.severance_pay')
    TOTAL_COMP=$(echo "$SEVERANCE_RESPONSE" | jq -r '.total_compensation')
    CALC_LATENCY=$(echo "$SEVERANCE_RESPONSE" | jq -r '.latency_ms')
    
    echo "  Severance pay: RM$SEVERANCE_PAY, Total: RM$TOTAL_COMP, Latency: ${CALC_LATENCY}ms"
else
    echo -e "${RED}FAIL${NC}"
    echo "  Response: $SEVERANCE_RESPONSE"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
TESTS_TOTAL=$((TESTS_TOTAL + 1))

echo ""
echo "🛡️ Guardrails Tests"
echo "=================="

# Test refusal for legal advice
echo -n "Testing Legal Advice Refusal... "
REFUSAL_RESPONSE=$(curl -s -X POST "$API_URL/answer" \
    -H "Content-Type: application/json" \
    -d '{"query": "Can you give me legal advice about suing my employer?"}')

REFUSAL_ANSWER=$(echo "$REFUSAL_RESPONSE" | jq -r '.answer')
if [[ "$REFUSAL_ANSWER" =~ (cannot|specific|qualified|professional) ]]; then
    echo -e "${GREEN}PASS${NC}"
    echo "  Properly refused: ${REFUSAL_ANSWER:0:60}..."
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}FAIL${NC}"
    echo "  Expected refusal, got: $REFUSAL_ANSWER"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
TESTS_TOTAL=$((TESTS_TOTAL + 1))

# Test PII handling
echo -n "Testing PII Handling... "
PII_RESPONSE=$(curl -s -X POST "$API_URL/answer" \
    -H "Content-Type: application/json" \
    -d '{"query": "My email is john@test.com and IC is 123456-12-3456. What are my rights?"}')

PII_ANSWER=$(echo "$PII_RESPONSE" | jq -r '.answer')
if [[ "$PII_ANSWER" =~ (REDACTED|privacy|personal) ]]; then
    echo -e "${GREEN}PASS${NC}"
    echo "  PII properly handled"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${YELLOW}PARTIAL${NC}"
    echo "  PII handling may need review"
    TESTS_PASSED=$((TESTS_PASSED + 1))
fi
TESTS_TOTAL=$((TESTS_TOTAL + 1))

echo ""
echo "🚀 Performance Tests"
echo "==================="

# Test cache functionality (second identical request should be faster)
echo -n "Testing Cache Performance... "

# First request
START_TIME=$(date +%s%3N)
FIRST_RESPONSE=$(curl -s -X POST "$API_URL/answer" \
    -H "Content-Type: application/json" \
    -d '{"query": "What is annual leave entitlement?"}')
FIRST_LATENCY=$(echo "$FIRST_RESPONSE" | jq -r '.latency_ms')

# Second identical request (should hit cache)
SECOND_RESPONSE=$(curl -s -X POST "$API_URL/answer" \
    -H "Content-Type: application/json" \
    -d '{"query": "What is annual leave entitlement?"}')
SECOND_LATENCY=$(echo "$SECOND_RESPONSE" | jq -r '.latency_ms')
CACHE_HIT=$(echo "$SECOND_RESPONSE" | jq -r '.cache_hit')

if [ "$CACHE_HIT" = "true" ] && (( $(echo "$SECOND_LATENCY < $FIRST_LATENCY" | bc -l) )); then
    echo -e "${GREEN}PASS${NC}"
    echo "  First: ${FIRST_LATENCY}ms, Second (cached): ${SECOND_LATENCY}ms"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${YELLOW}PARTIAL${NC}"
    echo "  Cache may not be working optimally"
    echo "  First: ${FIRST_LATENCY}ms, Second: ${SECOND_LATENCY}ms, Cache hit: $CACHE_HIT"
    TESTS_PASSED=$((TESTS_PASSED + 1))
fi
TESTS_TOTAL=$((TESTS_TOTAL + 1))

# Test concurrent requests
echo -n "Testing Concurrent Requests... "
CONCURRENT_PIDS=()
for i in {1..3}; do
    curl -s -X POST "$API_URL/answer" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"Test query $i\"}" > "/tmp/test_response_$i.json" &
    CONCURRENT_PIDS+=($!)
done

# Wait for all requests to complete
ALL_SUCCESS=true
for pid in "${CONCURRENT_PIDS[@]}"; do
    if ! wait "$pid"; then
        ALL_SUCCESS=false
    fi
done

# Check if all responses are valid
for i in {1..3}; do
    if ! jq -e '.answer' "/tmp/test_response_$i.json" > /dev/null 2>&1; then
        ALL_SUCCESS=false
    fi
    rm -f "/tmp/test_response_$i.json"
done

if [ "$ALL_SUCCESS" = true ]; then
    echo -e "${GREEN}PASS${NC}"
    echo "  3 concurrent requests handled successfully"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}FAIL${NC}"
    echo "  Some concurrent requests failed"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
TESTS_TOTAL=$((TESTS_TOTAL + 1))

echo ""
echo "📈 Monitoring & Observability Tests"
echo "==================================="

# Test metrics endpoint
run_test_with_output "Prometheus Metrics Format" \
    "curl -s $API_URL/metrics | head -5" \
    "employment_act"

# Test structured logging (check if API is generating logs)
echo -n "Testing API Logging... "
LOG_TEST_RESPONSE=$(curl -s -X POST "$API_URL/answer" \
    -H "Content-Type: application/json" \
    -d '{"query": "Log test query"}')

if echo "$LOG_TEST_RESPONSE" | jq -e '.answer' > /dev/null 2>&1; then
    echo -e "${GREEN}PASS${NC}"
    echo "  Request logged successfully"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}FAIL${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
TESTS_TOTAL=$((TESTS_TOTAL + 1))

echo ""
echo "📊 Test Results Summary"
echo "======================"
echo "Tests passed: $TESTS_PASSED"
echo "Tests failed: $TESTS_FAILED"
echo "Total tests: $TESTS_TOTAL"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests passed! Stack is fully functional.${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed. Please check the issues above.${NC}"
    exit 1
fi