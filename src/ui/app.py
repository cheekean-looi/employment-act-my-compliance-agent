#!/usr/bin/env python3
"""
Streamlit UI for Employment Act Malaysia Compliance Agent.
Features: chat interface, citations with modals, severance calculator, metrics display.
"""

import streamlit as st
import requests
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass


# Configuration
import os
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8001")
CHAT_HISTORY_KEY = "chat_history"
METRICS_KEY = "metrics_history"


@dataclass
class ChatMessage:
    """Chat message structure."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class QueryMetrics:
    """Query performance metrics."""
    timestamp: datetime
    latency_ms: float
    token_usage: Dict[str, int]
    cache_hit: bool
    confidence: float
    should_escalate: bool


def init_session_state():
    """Initialize Streamlit session state."""
    if CHAT_HISTORY_KEY not in st.session_state:
        st.session_state[CHAT_HISTORY_KEY] = []
    
    if METRICS_KEY not in st.session_state:
        st.session_state[METRICS_KEY] = []
    
    if "show_severance_calc" not in st.session_state:
        st.session_state.show_severance_calc = False
    
    if "selected_citation" not in st.session_state:
        st.session_state.selected_citation = None


def call_api(endpoint: str, method: str = "GET", json_data: Optional[Dict] = None) -> Dict[str, Any]:
    """Make API call with error handling."""
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "POST":
            response = requests.post(url, json=json_data, timeout=60)
        else:
            response = requests.get(url, timeout=30)
        
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API server. Please ensure the server is running.")
        return {"error": "Connection failed"}
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. Please try again.")
        return {"error": "Timeout"}
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ API error: {e.response.status_code}")
        return {"error": f"HTTP {e.response.status_code}"}
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return {"error": str(e)}


def display_chat_message(message: ChatMessage):
    """Display a chat message with appropriate styling."""
    if message.role == "user":
        with st.chat_message("user"):
            st.write(message.content)
    else:
        with st.chat_message("assistant"):
            st.write(message.content)
            
            # Display metadata if available
            if message.metadata:
                display_message_metadata(message.metadata)


def display_message_metadata(metadata: Dict[str, Any]):
    """Display message metadata (citations, metrics, flags)."""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Display citations
        citations = metadata.get("citations", [])
        if citations:
            st.write("**📚 Citations:**")
            for i, citation in enumerate(citations):
                if st.button(
                    f"📖 {citation['section_id']}", 
                    key=f"citation_{metadata.get('timestamp', time.time())}_{i}",
                    help="Click to view full section"
                ):
                    st.session_state.selected_citation = citation
                    st.rerun()
    
    with col2:
        # Display confidence and escalation
        confidence = metadata.get("confidence", 0)
        should_escalate = metadata.get("should_escalate", False)
        
        confidence_color = "green" if confidence > 0.8 else "orange" if confidence > 0.5 else "red"
        st.markdown(f"**🎯 Confidence:** <span style='color:{confidence_color}'>{confidence:.1%}</span>", 
                   unsafe_allow_html=True)
        
        if should_escalate:
            st.markdown("**⚠️ Escalation:** Required", help="This query requires human review")
    
    with col3:
        # Display performance metrics
        latency_ms = metadata.get("latency_ms", 0)
        cache_hit = metadata.get("cache_hit", False)
        token_usage = metadata.get("token_usage", {})
        
        cache_icon = "⚡" if cache_hit else "🔄"
        st.write(f"**{cache_icon} Latency:** {latency_ms:.0f}ms")
        
        if token_usage.get("total_tokens", 0) > 0:
            st.write(f"**🪙 Tokens:** {token_usage['total_tokens']}")


def display_citation_modal():
    """Display citation modal if citation is selected."""
    if st.session_state.selected_citation:
        citation = st.session_state.selected_citation
        
        # Create modal using columns and container
        st.markdown("---")
        st.subheader(f"📖 {citation['section_id']}")
        
        # Try to fetch full section content
        section_data = call_api(f"/section/{citation['section_id']}")
        
        if "error" not in section_data:
            st.write("**Full Section Text:**")
            st.text_area(
                "Section Content", 
                value=section_data.get("full_text", citation.get("snippet", "")),
                height=300,
                disabled=True
            )
            
            # Related sections
            related = section_data.get("related_sections", [])
            if related:
                st.write("**Related Sections:**")
                for related_section in related:
                    st.write(f"• {related_section}")
        else:
            st.write("**Snippet:**")
            st.text_area(
                "Citation Snippet",
                value=citation.get("snippet", "No content available"),
                height=150,
                disabled=True
            )
        
        if st.button("Close", key="close_citation"):
            st.session_state.selected_citation = None
            st.rerun()


def render_severance_calculator():
    """Render severance pay calculator."""
    st.subheader("💰 Severance Pay Calculator")
    
    with st.form("severance_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            monthly_wage = st.number_input(
                "Monthly Wage (MYR)", 
                min_value=1.0, 
                max_value=50000.0, 
                value=3000.0,
                step=100.0,
                help="Your gross monthly salary"
            )
            
            termination_reason = st.selectbox(
                "Termination Reason",
                options=[
                    "resignation", 
                    "dismissal_with_cause", 
                    "dismissal_without_cause",
                    "redundancy", 
                    "retirement", 
                    "contract_expiry", 
                    "mutual_agreement"
                ],
                format_func=lambda x: x.replace("_", " ").title(),
                help="Reason for employment termination"
            )
        
        with col2:
            years_of_service = st.number_input(
                "Years of Service", 
                min_value=0.0, 
                max_value=60.0, 
                value=2.5,
                step=0.1,
                help="Total years worked with the employer"
            )
            
            annual_leave_days = st.number_input(
                "Unused Annual Leave Days", 
                min_value=0, 
                max_value=365, 
                value=0,
                help="Number of unused annual leave days"
            )
        
        calculate_clicked = st.form_submit_button("Calculate Severance Pay", type="primary")
    
    if calculate_clicked:
        with st.spinner("Calculating severance pay..."):
            request_data = {
                "monthly_wage": monthly_wage,
                "years_of_service": years_of_service,
                "termination_reason": termination_reason,
                "annual_leave_days": annual_leave_days if annual_leave_days > 0 else None
            }
            
            result = call_api("/tool/severance", method="POST", json_data=request_data)
            
            if "error" not in result:
                # Display results
                st.success("✅ Calculation Complete")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Severance Pay", 
                        f"RM {result['severance_pay']:,.2f}",
                        help="Base severance payment"
                    )
                
                with col2:
                    if result.get("notice_pay"):
                        st.metric(
                            "Notice Pay", 
                            f"RM {result['notice_pay']:,.2f}",
                            help="Payment in lieu of notice"
                        )
                
                with col3:
                    st.metric(
                        "Total Compensation", 
                        f"RM {result['total_compensation']:,.2f}",
                        help="Total amount payable"
                    )
                
                # Display breakdown
                if result.get("calculation_breakdown"):
                    st.write("**📋 Calculation Breakdown:**")
                    breakdown = result["calculation_breakdown"]
                    
                    for key, value in breakdown.items():
                        if isinstance(value, (int, float)) and key != "severance_pay":
                            st.write(f"• {key.replace('_', ' ').title()}: RM {value:,.2f}")
                
                # Display legal references
                if result.get("employment_act_references"):
                    st.write("**⚖️ Legal References:**")
                    for ref in result["employment_act_references"]:
                        st.write(f"• {ref}")
                
                # Performance info
                latency = result.get("latency_ms", 0)
                st.caption(f"Calculated in {latency:.0f}ms")
            
            else:
                st.error(f"Calculation failed: {result['error']}")


def render_metrics_dashboard():
    """Render metrics and performance dashboard."""
    st.subheader("📊 Performance Metrics")
    
    # Get API health
    health = call_api("/health")
    
    if "error" not in health:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_color = "green" if health["status"] == "ok" else "red"
            st.markdown(f"**Status:** <span style='color:{status_color}'>{health['status'].upper()}</span>", 
                       unsafe_allow_html=True)
        
        with col2:
            st.write(f"**Uptime:** {health['uptime_s']/3600:.1f}h")
        
        with col3:
            vllm_icon = "✅" if health["vllm_ready"] else "❌"
            st.write(f"**vLLM:** {vllm_icon}")
        
        with col4:
            cache_icon = "✅" if health["cache_status"] == "ok" else "❌"
            st.write(f"**Cache:** {cache_icon}")
    
    # Show query metrics from session
    metrics_history = st.session_state[METRICS_KEY]
    
    if metrics_history:
        # Create charts
        df_data = []
        for metric in metrics_history[-20:]:  # Last 20 queries
            df_data.append({
                "timestamp": metric.timestamp,
                "latency_ms": metric.latency_ms,
                "tokens": metric.token_usage.get("total_tokens", 0),
                "confidence": metric.confidence,
                "cache_hit": metric.cache_hit
            })
        
        if df_data:
            import pandas as pd
            df = pd.DataFrame(df_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Latency chart
                fig_latency = px.line(
                    df, 
                    x="timestamp", 
                    y="latency_ms",
                    title="Response Latency",
                    markers=True
                )
                fig_latency.update_layout(height=300)
                st.plotly_chart(fig_latency, use_container_width=True)
            
            with col2:
                # Token usage chart
                fig_tokens = px.bar(
                    df, 
                    x="timestamp", 
                    y="tokens",
                    title="Token Usage",
                    color="cache_hit",
                    color_discrete_map={True: "green", False: "blue"}
                )
                fig_tokens.update_layout(height=300)
                st.plotly_chart(fig_tokens, use_container_width=True)


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Employment Act Malaysia - AI Assistant",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("⚖️ Employment Act Assistant")
        st.write("AI-powered guidance for Malaysian employment law")
        
        # Severance calculator toggle
        if st.button("💰 Severance Calculator", use_container_width=True):
            st.session_state.show_severance_calc = not st.session_state.show_severance_calc
        
        # Metrics toggle
        show_metrics = st.checkbox("📊 Show Metrics", value=False)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state[CHAT_HISTORY_KEY] = []
            st.session_state[METRICS_KEY] = []
            st.rerun()
        
        # API status
        st.markdown("---")
        st.write("**API Status**")
        health = call_api("/health")
        if "error" not in health:
            st.success(f"✅ {health['status'].upper()}")
            st.caption(f"Version: {health['version']}")
        else:
            st.error("❌ API Unavailable")
    
    # Main content area
    if st.session_state.show_severance_calc:
        render_severance_calculator()
        st.markdown("---")
    
    if show_metrics:
        render_metrics_dashboard()
        st.markdown("---")
    
    # Chat interface
    st.title("💬 Chat Assistant")
    
    # Display chat history
    chat_history = st.session_state[CHAT_HISTORY_KEY]
    
    for message in chat_history:
        display_chat_message(message)
    
    # Display citation modal if needed
    display_citation_modal()
    
    # Chat input
    if prompt := st.chat_input("Ask about Malaysian employment law..."):
        # Add user message
        user_message = ChatMessage(
            role="user",
            content=prompt,
            timestamp=datetime.now()
        )
        st.session_state[CHAT_HISTORY_KEY].append(user_message)
        
        # Display user message immediately
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start_time = time.time()
                
                request_data = {"query": prompt}
                response = call_api("/answer", method="POST", json_data=request_data)
                
                if "error" not in response:
                    # Display response
                    st.write(response["answer"])
                    
                    # Create assistant message with metadata
                    assistant_message = ChatMessage(
                        role="assistant",
                        content=response["answer"],
                        timestamp=datetime.now(),
                        metadata=response
                    )
                    st.session_state[CHAT_HISTORY_KEY].append(assistant_message)
                    
                    # Display metadata
                    display_message_metadata(response)
                    
                    # Record metrics
                    if response.get("token_usage"):
                        metrics = QueryMetrics(
                            timestamp=datetime.now(),
                            latency_ms=response.get("latency_ms", 0),
                            token_usage=response["token_usage"],
                            cache_hit=response.get("cache_hit", False),
                            confidence=response.get("confidence", 0),
                            should_escalate=response.get("should_escalate", False)
                        )
                        st.session_state[METRICS_KEY].append(metrics)
                        
                        # Keep only last 50 metrics
                        if len(st.session_state[METRICS_KEY]) > 50:
                            st.session_state[METRICS_KEY] = st.session_state[METRICS_KEY][-50:]
                
                else:
                    error_msg = f"Sorry, I encountered an error: {response['error']}"
                    st.error(error_msg)
                    
                    # Add error message to history
                    error_message = ChatMessage(
                        role="assistant",
                        content=error_msg,
                        timestamp=datetime.now()
                    )
                    st.session_state[CHAT_HISTORY_KEY].append(error_message)
        
        # Rerun to show the new messages
        st.rerun()


if __name__ == "__main__":
    main()