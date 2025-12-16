/**
 * Chat Application - Frontend Logic
 * Dynamically determines base URL to work with various hosting scenarios
 */

(function() {
    'use strict';

    // Dynamically determine the API base URL from current page location
    function getApiBaseUrl() {
        // Get the current page's path and remove the filename/trailing parts
        const pathname = window.location.pathname;
        
        // If we're at /some/prefix/index.html or /some/prefix/, get /some/prefix
        // If we're at /some/prefix (no trailing slash), get /some/prefix
        let basePath = pathname;
        
        // Remove trailing index.html if present
        if (basePath.endsWith('/index.html')) {
            basePath = basePath.slice(0, -11);
        }
        
        // Remove trailing slash if present (including root /)
        while (basePath.endsWith('/')) {
            basePath = basePath.slice(0, -1);
        }
        
        // Construct full base URL (no trailing slash)
        const baseUrl = `${window.location.protocol}//${window.location.host}${basePath}`;
        
        console.log('API Base URL:', baseUrl);
        return baseUrl;
    }

    const API_BASE = getApiBaseUrl();

    // DOM Elements
    const chatContainer = document.getElementById('chatContainer');
    const messageInput = document.getElementById('messageInput');
    const sendBtn = document.getElementById('sendBtn');
    const welcomeScreen = document.getElementById('welcomeScreen');

    // State
    let isFirstMessage = true;
    let conversationId = null;

    /**
     * Send a suggestion as a message
     */
    function sendSuggestion(text) {
        messageInput.value = text;
        sendMessage();
    }

    /**
     * Send the current message to the API
     */
    async function sendMessage() {
        const message = messageInput.value.trim();
        if (!message) return;

        // Hide welcome screen on first message
        if (isFirstMessage) {
            welcomeScreen.remove();
            isFirstMessage = false;
        }

        // Add user message to chat
        addMessage(message, 'user');
        messageInput.value = '';

        // Disable input while processing
        sendBtn.disabled = true;
        messageInput.disabled = true;

        // Show typing indicator
        const typingId = showTypingIndicator();

        try {
            const response = await fetch(`${API_BASE}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    conversation_id: conversationId
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to get response');
            }

            const data = await response.json();
            conversationId = data.conversation_id;

            // Remove typing indicator and add response
            removeTypingIndicator(typingId);
            addMessage(data.response, 'agent');

        } catch (error) {
            removeTypingIndicator(typingId);
            showError(error.message);
        } finally {
            sendBtn.disabled = false;
            messageInput.disabled = false;
            messageInput.focus();
        }
    }

    /**
     * Add a message to the chat container
     */
    function addMessage(content, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;

        const avatar = type === 'user' ? '👤' : '🤖';

        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">${escapeHtml(content)}</div>
        `;

        chatContainer.appendChild(messageDiv);
        scrollToBottom();
    }

    /**
     * Show typing indicator while waiting for response
     */
    function showTypingIndicator() {
        const id = 'typing-' + Date.now();
        const typingDiv = document.createElement('div');
        typingDiv.id = id;
        typingDiv.className = 'message agent';
        typingDiv.innerHTML = `
            <div class="message-avatar">🤖</div>
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;
        chatContainer.appendChild(typingDiv);
        scrollToBottom();
        return id;
    }

    /**
     * Remove typing indicator by ID
     */
    function removeTypingIndicator(id) {
        const element = document.getElementById(id);
        if (element) element.remove();
    }

    /**
     * Show an error message
     */
    function showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = `Error: ${message}`;
        chatContainer.appendChild(errorDiv);
        scrollToBottom();

        // Auto-remove after 5 seconds
        setTimeout(() => errorDiv.remove(), 5000);
    }

    /**
     * Escape HTML to prevent XSS
     */
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Scroll chat to bottom
     */
    function scrollToBottom() {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    // Event Listeners
    sendBtn.addEventListener('click', sendMessage);

    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Handle suggestion button clicks using event delegation
    document.addEventListener('click', (e) => {
        if (e.target.classList.contains('suggestion')) {
            const suggestion = e.target.dataset.suggestion;
            if (suggestion) {
                sendSuggestion(suggestion);
            }
        }
    });

    // Focus input on load
    messageInput.focus();

    // Log initialization
    console.log('Chat app initialized. API endpoint:', `${API_BASE}/chat`);

})();

