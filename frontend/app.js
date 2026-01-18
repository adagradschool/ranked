const tabs = document.querySelectorAll(".nav-item");
const panels = document.querySelectorAll(".tab-panel");
const toast = document.getElementById("toast");
const helpButton = document.getElementById("helpButton");
const gearButton = document.getElementById("gearButton");
const settingsButton = document.getElementById("settingsButton");
const avatarButton = document.getElementById("avatarButton");
const brandHome = document.getElementById("brandHome");

const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const chatMessages = document.getElementById("chatMessages");

const businessSearch = document.getElementById("businessSearch");
const businessDropdown = document.getElementById("businessDropdown");
const analyzeButton = document.getElementById("analyzeButton");

const fallbackBusinesses = [
  "Evergreen Dental Care",
  "Hudson Health Clinic",
  "Sunrise Pediatrics",
  "Westside Wellness",
  "Park Avenue Bistro",
  "East Village Cafe",
  "Chelsea Fitness Lab",
  "Brooklyn Family Dentistry",
];

function setActiveTab(tabId) {
  tabs.forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.tab === tabId);
  });
  panels.forEach((panel) => {
    panel.classList.toggle("active", panel.id === tabId);
  });
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  window.clearTimeout(showToast.timeoutId);
  showToast.timeoutId = window.setTimeout(() => {
    toast.classList.remove("show");
  }, 2200);
}

tabs.forEach((tab) => {
  tab.addEventListener("click", () => setActiveTab(tab.dataset.tab));
});

brandHome.addEventListener("click", () => setActiveTab("chat"));

helpButton.addEventListener("click", () => {
  showToast("Tips: Ask about a business, product, or location.");
});

gearButton.addEventListener("click", () => {
  setActiveTab("analytics");
  showToast("Analytics loaded. Pick a business to analyze.");
});

settingsButton.addEventListener("click", () => {
  showToast("Settings panel coming soon.");
});

avatarButton.addEventListener("click", () => {
  showToast("Signed in as Ada.");
});

function createMessage(role, text, pending = false, renderMarkdown = false) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  if (renderMarkdown && window.marked) {
    bubble.innerHTML = window.marked.parse(text);
  } else {
    bubble.textContent = text;
  }
  if (pending) {
    bubble.dataset.pending = "true";
  }
  wrapper.appendChild(bubble);
  chatMessages.appendChild(wrapper);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return bubble;
}

async function sendChat(question) {
  const response = await fetch("/rag-chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!response.ok) {
    throw new Error("Chat request failed.");
  }
  return response.json();
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = chatInput.value.trim();
  if (!question) {
    return;
  }
  chatInput.value = "";
  createMessage("user", question);
  const pendingBubble = createMessage("assistant", "Thinking...", true);
  try {
    const data = await sendChat(question);
    const finalText = `${data.answer}`.trim();
    if (window.marked) {
      pendingBubble.innerHTML = window.marked.parse(finalText);
    } else {
      pendingBubble.textContent = finalText;
    }
  } catch (error) {
    pendingBubble.textContent = "I could not reach the RAG service. Try again.";
  }
});

async function fetchBusinesses(query) {
  const response = await fetch(`/businesses?q=${encodeURIComponent(query)}&limit=8`);
  if (!response.ok) {
    throw new Error("Business lookup failed.");
  }
  const data = await response.json();
  return data.businesses || [];
}

function renderDropdown(items) {
  businessDropdown.innerHTML = "";
  if (!items.length) {
    businessDropdown.style.display = "none";
    return;
  }
  items.forEach((item) => {
    const option = document.createElement("button");
    option.type = "button";
    option.textContent = item;
    option.addEventListener("click", () => {
      businessSearch.value = item;
      businessDropdown.style.display = "none";
    });
    businessDropdown.appendChild(option);
  });
  businessDropdown.style.display = "flex";
}

let businessTimer;
businessSearch.addEventListener("input", () => {
  clearTimeout(businessTimer);
  const query = businessSearch.value.trim();
  businessTimer = setTimeout(async () => {
    if (!query) {
      renderDropdown([]);
      return;
    }
    try {
      const results = await fetchBusinesses(query);
      renderDropdown(results.length ? results : fallbackBusinesses);
    } catch (error) {
      renderDropdown(fallbackBusinesses);
    }
  }, 200);
});

businessSearch.addEventListener("blur", () => {
  setTimeout(() => {
    businessDropdown.style.display = "none";
  }, 150);
});

analyzeButton.addEventListener("click", () => {
  const query = businessSearch.value.trim() || "your business";
  createMessage(
    "assistant",
    `Running Ranked Analytics for ${query}. Results will appear here once the data pipeline is connected.`
  );
  setActiveTab("chat");
});
