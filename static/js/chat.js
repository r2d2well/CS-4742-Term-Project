let sessionId = null;
let firstSubmission = true;
const userTextInput = document.getElementById('userText');
const sendBtn = document.getElementById('sendBtn');
const resetBtn = document.getElementById('resetBtn');
const chatArea = document.getElementById('chatArea');
const sessionInfoDiv = document.getElementById('sessionInfo');
const chatContainer = document.getElementById('chatContainer');
let isFirstMessage = true;

userTextInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') {
        e.preventDefault();
        sendBtn.click();
    }
});

sendBtn.addEventListener('click', () => {
    const text = userTextInput.value.trim();
    if (!text) return;

    if (firstSubmission) {
        chatContainer.classList.remove('minimized');
        chatContainer.classList.add('expanded');
        firstSubmission = false;
    }

    if (text.toLowerCase() === 'reset') {
        if (sessionId) {
            resetSession(sessionId);
        } else {
            addBotMessage("No active session to reset. Please start by describing yourself.");
        }
        userTextInput.value = '';
        return;
    }

    addUserMessage(text);
    userTextInput.value = '';

    if (!sessionId) {
        startSession(text);
    } else {
        continueSession(sessionId, text);
    }
});

resetBtn.addEventListener('click', () => {
    if (sessionId) {
        resetSession(sessionId);
    } else {
        addBotMessage("No active session. Please start by describing yourself.");
    }
});

function startSession(text) {
    fetch('/start', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: text})
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            addBotMessage(`Error: ${data.error}`);
            return;
        }

        sessionId = data.session_id;
        resetBtn.classList.remove('hidden');
        addBotMessage(formatMbtiMessage(data.mbti_type, data.explanation, data.image_url, data.message));
        sessionInfoDiv.textContent = `Current Session ID: ${sessionId}`;
    })
    .catch(err => addBotMessage(`Error starting session: ${err}`));
}

function continueSession(session_id, text) {
    fetch('/continue', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({session_id, text})
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            addBotMessage(`Error: ${data.error}`);
            return;
        }

        addBotMessage(formatMbtiMessage(data.mbti_type, data.explanation, data.image_url, data.message, true));
    })
    .catch(err => addBotMessage(`Error continuing session: ${err}`));
}

function resetSession(session_id) {
    fetch('/reset', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({session_id})
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            addBotMessage(`Error: ${data.error}`);
            return;
        }

        addBotMessage(data.message);
        sessionId = null;
        resetBtn.classList.add('hidden');
        sessionInfoDiv.textContent = 'Please start by describing yourself below...';
    })
    .catch(err => addBotMessage(`Error resetting session: ${err}`));
}

function addUserMessage(text) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message-bubble', 'user-message');
    msgDiv.textContent = text;
    chatArea.appendChild(msgDiv);
    scrollToBottom();
}

function addBotMessage(html) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message-bubble', 'bot-message');
    msgDiv.innerHTML = html;
    chatArea.appendChild(msgDiv);
    scrollToBottom();
}

function scrollToBottom() {
    chatArea.scrollTop = chatArea.scrollHeight;
}

function formatMbtiMessage(type, explanation, imageUrl, message, updated=false) {
    const titleText = updated ? "Updated MBTI Type" : "Predicted MBTI Type";
    return `
        <b>${titleText}:</b> <span style="font-size:1.2rem;color:var(--accent-color);font-weight:700;">${type}</span><br>
        <img src="${imageUrl}" alt="${type} image" style="max-width:200px;border-radius:8px;margin:1rem 0;">
        <p style="margin-top:1rem;">${explanation}</p>
        <p style="margin-top:1rem;">${message}</p>
    `;
}

function maximizeChatBox() {
    chatContainer.classList.remove('minimized');
    chatContainer.classList.add('maximized');
}
