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
const analysisStatus = document.getElementById("analysisStatus");
const analysisPercent = document.getElementById("analysisPercent");
const analysisProgress = document.getElementById("analysisProgress");
const analysisSteps = document.getElementById("analysisSteps");
const analysisScore = document.getElementById("analysisScore");
const analysisPresence = document.getElementById("analysisPresence");
const analysisAvgRank = document.getElementById("analysisAvgRank");
const analysisQueries = document.getElementById("analysisQueries");
const analysisBusiness = document.getElementById("analysisBusiness");
const analysisCompletion = document.getElementById("analysisCompletion");
const analysisTable = document.getElementById("analysisTable");
const analysisResults = document.getElementById("analysisResults");
const analysisEmpty = document.getElementById("analysisEmpty");
const analysisCompetitors = document.getElementById("analysisCompetitors");
const analysisCompetitorBase = document.getElementById("analysisCompetitorBase");
const blogStatus = document.getElementById("blogStatus");
const blogContent = document.getElementById("blogContent");
const blogReindexButton = document.getElementById("blogReindexButton");
const blogResetButton = document.getElementById("blogResetButton");

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

let analyticsJobId = null;
let analyticsPolling = null;
let analyticsBusy = false;
let lastBlog = null;

function setProgress(progress, stage, steps) {
  const safeProgress = Math.max(0, Math.min(100, progress || 0));
  analysisProgress.style.width = `${safeProgress}%`;
  analysisPercent.textContent = `${Math.round(safeProgress)}%`;
  analysisStatus.textContent = stage || "Working on analysis...";
  renderSteps(steps || []);
}

function renderSteps(steps) {
  analysisSteps.innerHTML = "";
  if (!steps.length) {
    return;
  }
  steps.forEach((step) => {
    const item = document.createElement("div");
    item.className = `progress-step ${step.status || ""}`.trim();
    const label = document.createElement("div");
    label.textContent = step.label || "Step";
    const status = document.createElement("span");
    status.textContent = step.status || "pending";
    item.appendChild(label);
    item.appendChild(status);
    analysisSteps.appendChild(item);
  });
}

function resetResults() {
  analysisScore.textContent = "--";
  analysisPresence.textContent = "--";
  analysisAvgRank.textContent = "--";
  analysisQueries.textContent = "--";
  analysisBusiness.textContent = "--";
  analysisCompletion.textContent = "--";
  analysisTable.innerHTML = "";
  analysisCompetitors.innerHTML = "";
  analysisCompetitorBase.textContent = "--";
  blogStatus.textContent = "Pick a question to generate.";
  blogContent.innerHTML = "";
  lastBlog = null;
  blogReindexButton.disabled = true;
  blogResetButton.disabled = true;
  analysisResults.classList.remove("active");
  analysisEmpty.style.display = "block";
}

function renderResults(result) {
  const scoreValue =
    typeof result.score === "number" ? result.score.toFixed(1) : result.score;
  const presenceValue =
    typeof result.presence_rate === "number"
      ? result.presence_rate.toFixed(1)
      : result.presence_rate;
  analysisScore.textContent = scoreValue;
  analysisPresence.textContent = `${presenceValue}%`;
  analysisAvgRank.textContent = result.average_rank
    ? `#${Number(result.average_rank).toFixed(1)}`
    : "--";
  analysisQueries.textContent = `${result.total_queries}`;
  analysisBusiness.textContent = result.matched_business_name || result.business_name;
  analysisCompletion.textContent = "Completed";
  analysisCompetitorBase.textContent = `Your score ${scoreValue}`;

  analysisTable.innerHTML = "";
  const header = document.createElement("div");
  header.className = "results-row header";
  ["Query", "Found", "Rank", "Top businesses", "Blog"].forEach((text) => {
    const cell = document.createElement("div");
    cell.textContent = text;
    header.appendChild(cell);
  });
  analysisTable.appendChild(header);

  result.queries.forEach((item) => {
    const row = document.createElement("div");
    row.className = "results-row";

    const queryCell = document.createElement("div");
    queryCell.className = "results-cell";
    queryCell.textContent = item.query;

    const foundCell = document.createElement("div");
    foundCell.className = "results-cell";
    const foundPill = document.createElement("span");
    foundPill.className = `rank-pill ${item.found ? "" : "missed"}`.trim();
    foundPill.textContent = item.found ? "Yes" : "No";
    foundCell.appendChild(foundPill);

    const rankCell = document.createElement("div");
    rankCell.className = "results-cell";
    const rankPill = document.createElement("span");
    rankPill.className = `rank-pill ${item.rank ? "" : "missed"}`.trim();
    rankPill.textContent = item.rank ? `#${item.rank}` : "--";
    rankCell.appendChild(rankPill);

    const topCell = document.createElement("div");
    topCell.className = "results-cell";
    const tagList = document.createElement("div");
    tagList.className = "tag-list";
    const topList = (item.top_businesses || []).slice(0, 5);
    topList.forEach((biz) => {
      const label = biz.business_name || biz.business_url || "Unknown";
      if (label.toLowerCase() === "unknown") {
        return;
      }
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = `#${biz.rank} ${label}`;
      tagList.appendChild(tag);
    });
    if (!tagList.children.length) {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = "No results";
      tagList.appendChild(tag);
    }
    topCell.appendChild(tagList);

    row.appendChild(queryCell);
    row.appendChild(foundCell);
    row.appendChild(rankCell);
    row.appendChild(topCell);
    const blogCell = document.createElement("div");
    blogCell.className = "results-cell";
    const blogButton = document.createElement("button");
    blogButton.type = "button";
    blogButton.className = "blog-button";
    blogButton.textContent = "Add Blog Article";
    blogButton.addEventListener("click", () => generateBlogArticle(item.query, blogButton));
    blogCell.appendChild(blogButton);
    row.appendChild(blogCell);
    analysisTable.appendChild(row);
  });

  analysisEmpty.style.display = "none";
  analysisResults.classList.add("active");

  renderScoreList(analysisCompetitors, result.top_competitors || [], true);
}

async function generateBlogArticle(question, button) {
  if (!question) {
    return;
  }
  const businessName = analysisBusiness.textContent || businessSearch.value.trim();
  if (!businessName) {
    showToast("Pick a business first.");
    return;
  }
  if (button) {
    button.disabled = true;
    button.classList.add("loading");
    button.textContent = "Generating...";
  }
  blogStatus.textContent = "Generating SEO article...";
  blogContent.innerHTML = "";
  try {
    const response = await fetch("/generate-blog", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ business_name: businessName, question }),
    });
    if (!response.ok) {
      throw new Error("Blog generation failed.");
    }
    const data = await response.json();
    if (data.html) {
      const parser = new DOMParser();
      const doc = parser.parseFromString(data.html, "text/html");
      blogContent.innerHTML = doc.body ? doc.body.innerHTML : data.html;
      blogContent.scrollIntoView({ behavior: "smooth", block: "start" });
    } else {
      blogContent.textContent = "No article returned.";
    }
    lastBlog = {
      businessName,
      question,
      html: data.html || "",
      blogUrl: data.blog_url || "",
    };
    blogReindexButton.disabled = !lastBlog.html || !lastBlog.blogUrl;
    blogResetButton.disabled = !lastBlog.blogUrl;
    blogStatus.textContent = "Generated.";
  } catch (error) {
    blogStatus.textContent = "Failed to generate.";
    showToast("Could not generate blog article.");
  } finally {
    if (button) {
      button.disabled = false;
      button.classList.remove("loading");
      button.textContent = "Add Blog Article";
    }
  }
}

function renderScoreList(container, items, showBar) {
  container.innerHTML = "";
  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "meta-value";
    empty.textContent = "No competitors found.";
    container.appendChild(empty);
    return;
  }
  items.forEach((item) => {
    const scoreValue =
      typeof item.visibility_score === "number"
        ? item.visibility_score
        : Number(item.visibility_score) || 0;
    const label = item.business_name || item.business_url || "Unknown";
    if (label.toLowerCase() === "unknown") {
      return;
    }
    const row = document.createElement("div");
    row.className = "competitor-item";

    const name = document.createElement("div");
    name.className = "competitor-name";
    name.textContent = label;

    if (showBar) {
      const bar = document.createElement("div");
      bar.className = "score-bar";
      const fill = document.createElement("div");
      fill.className = "score-bar-fill";
      fill.style.width = `${scoreValue}%`;
      bar.appendChild(fill);
      row.appendChild(name);
      row.appendChild(bar);
    } else {
      row.appendChild(name);
    }

    const score = document.createElement("div");
    score.className = "competitor-score";
    score.textContent = scoreValue.toFixed(1);
    row.appendChild(score);

    container.appendChild(row);
  });
}

function stopPolling() {
  if (analyticsPolling) {
    clearInterval(analyticsPolling);
    analyticsPolling = null;
  }
}

async function pollAnalytics(jobId) {
  try {
    const response = await fetch(`/analytics/status/${jobId}`);
    if (!response.ok) {
      throw new Error("Status check failed.");
    }
    const data = await response.json();
    setProgress(data.progress, data.stage, data.steps);
    if (data.status === "completed" && data.result) {
      renderResults(data.result);
      showToast("Analytics complete.");
      stopPolling();
      analyticsBusy = false;
      analyzeButton.disabled = false;
      analyzeButton.textContent = "Analyze";
    }
    if (data.status === "failed") {
      showToast(data.error || "Analytics failed.");
      analysisCompletion.textContent = "Failed";
      analysisEmpty.style.display = "block";
      stopPolling();
      analyticsBusy = false;
      analyzeButton.disabled = false;
      analyzeButton.textContent = "Analyze";
    }
  } catch (error) {
    showToast("Could not reach analytics service.");
    analysisEmpty.style.display = "block";
    stopPolling();
    analyticsBusy = false;
    analyzeButton.disabled = false;
    analyzeButton.textContent = "Analyze";
  }
}

analyzeButton.addEventListener("click", async () => {
  if (analyticsBusy) {
    return;
  }
  const query = businessSearch.value.trim();
  if (!query) {
    showToast("Pick a business to analyze.");
    return;
  }
  analyticsBusy = true;
  analyzeButton.disabled = true;
  analyzeButton.textContent = "Analyzing...";
  resetResults();
  analysisCompletion.textContent = "Running";
  analysisEmpty.style.display = "none";
  setProgress(0, "Starting analysis...", buildDefaultSteps());

  try {
    const response = await fetch("/analytics/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ business_name: query }),
    });
    if (!response.ok) {
      throw new Error("Start failed.");
    }
    const data = await response.json();
    analyticsJobId = data.job_id;
    setProgress(data.progress, data.stage, data.steps);
    stopPolling();
    analyticsPolling = setInterval(() => pollAnalytics(analyticsJobId), 1200);
  } catch (error) {
    showToast("Analytics could not start.");
    analysisEmpty.style.display = "block";
    analyticsBusy = false;
    analyzeButton.disabled = false;
    analyzeButton.textContent = "Analyze";
  }
});

function buildDefaultSteps() {
  return [
    { label: "Load business content", status: "pending" },
    { label: "Generate search questions", status: "pending" },
    { label: "Querying LLMs", status: "pending" },
    { label: "Compute visibility score", status: "pending" },
  ];
}

async function reindexBlog() {
  if (!lastBlog || !lastBlog.html || !lastBlog.blogUrl) {
    showToast("Generate a blog first.");
    return;
  }
  blogReindexButton.disabled = true;
  blogReindexButton.classList.add("loading");
  blogStatus.textContent = "Reindexing blog...";
  try {
    const response = await fetch("/reindex-blog", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        html: lastBlog.html,
        blog_url: lastBlog.blogUrl,
        business_name: lastBlog.businessName,
        question: lastBlog.question,
      }),
    });
    if (!response.ok) {
      throw new Error("Reindex failed.");
    }
    const data = await response.json();
    blogStatus.textContent = `Reindexed (${data.chunks_indexed} chunks).`;
    showToast("Blog reindexed.");
  } catch (error) {
    blogStatus.textContent = "Reindex failed.";
    showToast("Could not reindex blog.");
  } finally {
    blogReindexButton.disabled = false;
    blogReindexButton.classList.remove("loading");
  }
}

async function resetBlog() {
  if (!lastBlog || !lastBlog.blogUrl) {
    showToast("Nothing to reset.");
    return;
  }
  blogResetButton.disabled = true;
  blogResetButton.classList.add("loading");
  blogStatus.textContent = "Resetting blog...";
  try {
    const response = await fetch("/delete-blog", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ blog_url: lastBlog.blogUrl }),
    });
    if (!response.ok) {
      throw new Error("Reset failed.");
    }
    const data = await response.json();
    blogStatus.textContent = `Reset (${data.chunks_deleted} chunks removed).`;
    blogContent.innerHTML = "";
    lastBlog = null;
    blogReindexButton.disabled = true;
    blogResetButton.disabled = true;
    showToast("Blog reset.");
  } catch (error) {
    blogStatus.textContent = "Reset failed.";
    showToast("Could not reset blog.");
  } finally {
    blogResetButton.classList.remove("loading");
  }
}

blogReindexButton.addEventListener("click", reindexBlog);
blogResetButton.addEventListener("click", resetBlog);
