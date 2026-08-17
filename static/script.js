// CityEye — Real-Time AI Video Detection & CCTV Analytics Script

document.addEventListener("DOMContentLoaded", () => {
  // DOM Header & Status Elements
  const liveClockEl = document.getElementById("liveClock");
  const alertBanner = document.getElementById("alertBanner");
  const alertTitle = document.getElementById("alertTitle");
  const alertMessage = document.getElementById("alertMessage");
  const alertCloseBtn = document.getElementById("alertCloseBtn");

  const streamStatusBadge = document.getElementById("streamStatusBadge");
  const streamDot = document.getElementById("streamDot");
  const streamStatusText = document.getElementById("streamStatusText");
  const cameraNameText = document.getElementById("cameraNameText");
  const aiEngineText = document.getElementById("aiEngineText");

  // Counter Elements
  const statTotalDetections = document.getElementById("statTotalDetections");
  const statTotalMeta = document.getElementById("statTotalMeta");
  const statPersons = document.getElementById("statPersons");
  const statCars = document.getElementById("statCars");
  const statBuses = document.getElementById("statBuses");
  const statTrucks = document.getElementById("statTrucks");
  const statMotorcycles = document.getElementById("statMotorcycles");

  // Violation Counters
  const statTriple = document.getElementById("statTriple");
  const statWrongWay = document.getElementById("statWrongWay");
  const statStopped = document.getElementById("statStopped");
  const statHelmet = document.getElementById("statHelmet");
  const eventTotalBadge = document.getElementById("eventTotalBadge");

  // Toolbar & Control Elements
  const videoSelect = document.getElementById("videoSelect");
  const confSlider = document.getElementById("confSlider");
  const confValue = document.getElementById("confValue");
  const loopToggle = document.getElementById("loopToggle");

  const btnStartDetection = document.getElementById("btnStartDetection");
  const btnStopDetection = document.getElementById("btnStopDetection");
  const btnPlaceholderStart = document.getElementById("btnPlaceholderStart");
  const btnRunProcess = document.getElementById("btnRunProcess");
  const btnRefresh = document.getElementById("btnRefresh");

  // Video & Stream Elements
  const videoFrameContainer = document.getElementById("videoFrameContainer");
  const liveVideoFeed = document.getElementById("liveVideoFeed");
  const processedVideoPlayer = document.getElementById("processedVideoPlayer");
  const videoPlaceholder = document.getElementById("videoPlaceholder");
  const videoResolutionBadge = document.getElementById("videoResolutionBadge");
  const fpsBadge = document.getElementById("fpsBadge");
  const frameBadge = document.getElementById("frameBadge");
  const liveRecDot = document.getElementById("liveRecDot");

  // Video Player Controls Elements (Play/Pause & Fullscreen)
  const btnPlayPause = document.getElementById("btnPlayPause");
  const iconPlay = document.getElementById("iconPlay");
  const iconPause = document.getElementById("iconPause");
  const labelPlayPause = document.getElementById("labelPlayPause");
  const btnFullscreen = document.getElementById("btnFullscreen");
  const iconFullscreen = document.getElementById("iconFullscreen");
  const labelFullscreen = document.getElementById("labelFullscreen");

  // Live Detection Panel Elements
  const liveDetectionsContainer = document.getElementById("liveDetectionsContainer");
  const liveDetectionsCountChip = document.getElementById("liveDetectionsCountChip");
  const liveStreamMeta = document.getElementById("liveStreamMeta");

  // Event Log Table Elements
  const eventsTableBody = document.getElementById("eventsTableBody");
  const eventFilterSelect = document.getElementById("eventFilterSelect");

  // State Variables
  let isStreaming = false;
  let telemetryInterval = null;
  let availableVideos = [];
  let allEvents = [];
  let seenEventKeys = new Set();

  // Class Icon & Badge Mapping
  const CLASS_ICONS = {
    car: "🚗",
    person: "🚶",
    motorcycle: "🏍️",
    bus: "🚌",
    truck: "🚚"
  };

  // -------------------------------------------------------------
  // 1. Live UTC Clock
  // -------------------------------------------------------------
  function updateClock() {
    const now = new Date();
    liveClockEl.textContent = now.toTimeString().split(" ")[0] + " " + now.toLocaleDateString();
  }
  setInterval(updateClock, 1000);
  updateClock();

  // -------------------------------------------------------------
  // 2. Alert Notification Banner
  // -------------------------------------------------------------
  function showAlert(title, message, type = "warning") {
    alertBanner.className = `alert-banner ${type}`;
    alertTitle.textContent = title;
    alertMessage.textContent = message;
    alertBanner.classList.remove("hidden");
  }

  alertCloseBtn.addEventListener("click", () => {
    alertBanner.classList.add("hidden");
  });

  // -------------------------------------------------------------
  // 3. Load Available Videos from Backend
  // -------------------------------------------------------------
  async function loadVideosList() {
    try {
      const res = await fetch("/videos");
      const data = await res.json();

      availableVideos = data.videos || [];
      videoSelect.innerHTML = "";

      if (availableVideos.length === 0) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "No videos found in videos/";
        videoSelect.appendChild(opt);
        return;
      }

      availableVideos.forEach(v => {
        const opt = document.createElement("option");
        opt.value = v.filename;
        opt.textContent = `${v.filename} (${v.frames} frames • ${v.resolution})`;
        videoSelect.appendChild(opt);
      });

      if (data.default_video) {
        videoSelect.value = data.default_video;
      }

      updateSelectedVideoMetadata();
    } catch (err) {
      console.error("Error loading video list:", err);
      showAlert("Video Load Error", "Could not fetch videos from backend.", "error");
    }
  }

  function updateSelectedVideoMetadata() {
    const selected = videoSelect.value;
    const v = availableVideos.find(item => item.filename === selected);
    if (v) {
      videoResolutionBadge.textContent = v.resolution || "720P HD";
      cameraNameText.textContent = `CAM: ${v.filename.split('.')[0].toUpperCase()}`;
    }
  }

  videoSelect.addEventListener("change", () => {
    updateSelectedVideoMetadata();
    if (isStreaming) {
      startLiveStream();
    }
  });

  // Confidence Slider Listener
  confSlider.addEventListener("input", () => {
    confValue.textContent = `${confSlider.value}%`;
  });

  confSlider.addEventListener("change", () => {
    if (isStreaming) {
      startLiveStream(); // Reload stream with updated confidence threshold
    }
  });

  // -------------------------------------------------------------
  // 4. Start Live Real-Time YOLO Video Detection
  // -------------------------------------------------------------
  function startLiveStream() {
    const selectedVideo = videoSelect.value;
    if (!selectedVideo) {
      showAlert("No Video Selected", "Please select an MP4 video from the dropdown.", "warning");
      return;
    }

    const conf = (parseInt(confSlider.value, 10) / 100).toFixed(2);
    const isLoop = loopToggle.checked;
    const timestamp = Date.now();

    // Prepare stream URL
    const streamUrl = `/video-feed?video=${encodeURIComponent(selectedVideo)}&conf=${conf}&loop=${isLoop}&_t=${timestamp}`;

    // Switch UI to active streaming
    isStreaming = true;
    liveVideoFeed.src = streamUrl;
    liveVideoFeed.classList.remove("hidden");
    processedVideoPlayer.classList.add("hidden");
    videoPlaceholder.classList.add("hidden");

    btnStartDetection.classList.add("hidden");
    btnStopDetection.classList.remove("hidden");

    // Update Player Control Button (Pause state)
    if (iconPlay && iconPause) {
      iconPlay.classList.add("hidden");
      iconPause.classList.remove("hidden");
    }
    if (labelPlayPause) labelPlayPause.textContent = "Pause";
    if (btnPlayPause) btnPlayPause.title = "Pause Video";

    streamStatusBadge.className = "status-badge";
    streamDot.className = "status-dot active";
    streamStatusText.textContent = "FEED: STREAMING";

    if (liveRecDot) {
      liveRecDot.style.display = "inline-block";
    }

    showAlert("Detection Started", `Real-time YOLO detection active on ${selectedVideo}.`, "success");

    // Start fast polling for telemetry (every 200ms)
    if (telemetryInterval) clearInterval(telemetryInterval);
    telemetryInterval = setInterval(fetchLiveTelemetry, 200);
  }

  // -------------------------------------------------------------
  // 5. Stop Live Video Detection
  // -------------------------------------------------------------
  async function stopLiveStream() {
    isStreaming = false;
    if (telemetryInterval) {
      clearInterval(telemetryInterval);
      telemetryInterval = null;
    }

    try {
      await fetch("/stop-feed", { method: "POST" });
    } catch (e) {
      console.warn("Stop feed signal error:", e);
    }

    // Reset visual feed
    liveVideoFeed.src = "";
    liveVideoFeed.classList.add("hidden");
    videoPlaceholder.classList.remove("hidden");

    btnStartDetection.classList.remove("hidden");
    btnStopDetection.classList.add("hidden");

    // Update Player Control Button (Play state)
    if (iconPlay && iconPause) {
      iconPlay.classList.remove("hidden");
      iconPause.classList.add("hidden");
    }
    if (labelPlayPause) labelPlayPause.textContent = "Play";
    if (btnPlayPause) btnPlayPause.title = "Play Video";

    streamDot.className = "status-dot";
    streamStatusText.textContent = "FEED: STOPPED";

    // Clear live detection cards
    liveDetectionsContainer.innerHTML = `
      <div class="empty-detections-state">
        <span>Video detection stopped. Click "Start Video Detection" or Play to resume.</span>
      </div>
    `;
    liveDetectionsCountChip.textContent = "0 Active";
    liveStreamMeta.textContent = "Feed idle";
  }

  btnStartDetection.addEventListener("click", startLiveStream);
  btnPlaceholderStart.addEventListener("click", startLiveStream);
  btnStopDetection.addEventListener("click", stopLiveStream);

  // Play / Pause Button Listener
  if (btnPlayPause) {
    btnPlayPause.addEventListener("click", () => {
      if (isStreaming) {
        stopLiveStream();
      } else {
        startLiveStream();
      }
    });
  }

  // Full Screen Button & Browser Fullscreen API
  function toggleFullscreen() {
    const targetElem = videoFrameContainer || document.getElementById("videoFrameContainer");
    if (!targetElem) return;

    const isFull = !!(
      document.fullscreenElement ||
      document.webkitFullscreenElement ||
      document.mozFullScreenElement ||
      document.msFullscreenElement
    );

    if (!isFull) {
      if (targetElem.requestFullscreen) {
        targetElem.requestFullscreen();
      } else if (targetElem.webkitRequestFullscreen) {
        targetElem.webkitRequestFullscreen();
      } else if (targetElem.mozRequestFullScreen) {
        targetElem.mozRequestFullScreen();
      } else if (targetElem.msRequestFullscreen) {
        targetElem.msRequestFullscreen();
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen();
      } else if (document.webkitExitFullscreen) {
        document.webkitExitFullscreen();
      } else if (document.mozCancelFullScreen) {
        document.mozCancelFullScreen();
      } else if (document.msExitFullscreen) {
        document.msExitFullscreen();
      }
    }
  }

  if (btnFullscreen) {
    btnFullscreen.addEventListener("click", toggleFullscreen);
  }

  function handleFullscreenChange() {
    const isFull = !!(
      document.fullscreenElement ||
      document.webkitFullscreenElement ||
      document.mozFullScreenElement ||
      document.msFullscreenElement
    );
    if (labelFullscreen) {
      labelFullscreen.textContent = isFull ? "Exit Full Screen" : "Full Screen";
    }
    if (btnFullscreen) {
      btnFullscreen.title = isFull ? "Exit Full Screen" : "Full Screen";
    }
  }

  document.addEventListener("fullscreenchange", handleFullscreenChange);
  document.addEventListener("webkitfullscreenchange", handleFullscreenChange);
  document.addEventListener("mozfullscreenchange", handleFullscreenChange);
  document.addEventListener("MSFullscreenChange", handleFullscreenChange);

  // -------------------------------------------------------------
  // 6. Fast Telemetry Polling (`/live-data`)
  // -------------------------------------------------------------
  async function fetchLiveTelemetry() {
    if (!isStreaming) return;

    try {
      const res = await fetch("/live-data");
      if (!res.ok) return;

      const data = await res.json();
      if (!data) return;

      // Update HUD badges
      if (data.fps) fpsBadge.textContent = `${data.fps} FPS`;
      if (data.frame_no !== undefined) {
        frameBadge.textContent = `FRAME: ${data.frame_no}${data.total_frames ? '/' + data.total_frames : ''}`;
      }

      // Update Live Counters
      const counts = data.counts || {};
      statTotalDetections.textContent = counts.total || 0;
      statTotalMeta.textContent = `Active in Frame ${data.frame_no || 0}`;
      statPersons.textContent = counts.person || 0;
      statCars.textContent = counts.car || 0;
      statBuses.textContent = counts.bus || 0;
      statTrucks.textContent = counts.truck || 0;
      statMotorcycles.textContent = counts.motorcycle || 0;

      // Update violation metrics
      if (counts.cumulative_events !== undefined) {
        eventTotalBadge.textContent = `${counts.cumulative_events} Events`;
      }

      // Render Live Detection Cards
      renderLiveDetectionCards(data.current_detections || [], data.frame_no || 0);

      // Ingest recent events
      if (data.recent_events && data.recent_events.length > 0) {
        data.recent_events.forEach(ev => {
          const evKey = `${ev.timestamp}_${ev.event}_${ev.vehicle_id}`;
          if (!seenEventKeys.has(evKey)) {
            seenEventKeys.add(evKey);
            allEvents.unshift(ev);
          }
        });
        renderEventsTable();
      }

    } catch (err) {
      console.warn("Telemetry fetch error:", err);
    }
  }

  // -------------------------------------------------------------
  // 7. Render Live Detection Cards (Current Frame Objects)
  // -------------------------------------------------------------
  function renderLiveDetectionCards(detections, frameNo) {
    liveDetectionsCountChip.textContent = `${detections.length} Active`;
    liveStreamMeta.textContent = `Frame ${frameNo} • ${detections.length} objects detected in real time`;

    if (detections.length === 0) {
      liveDetectionsContainer.innerHTML = `
        <div class="empty-detections-state">
          <span>No objects detected in current frame (Frame ${frameNo}).</span>
        </div>
      `;
      return;
    }

    liveDetectionsContainer.innerHTML = "";

    detections.forEach(d => {
      const card = document.createElement("div");
      card.className = "detection-card";

      const cname = (d.class_name || "object").toLowerCase();
      const icon = CLASS_ICONS[cname] || "📦";
      const confPct = d.confidence_pct || (d.confidence ? `${Math.round(d.confidence * 100)}%` : "90%");
      const confVal = d.confidence ? Math.round(d.confidence * 100) : 90;
      const trackId = d.track_id !== null && d.track_id !== undefined ? `#${d.track_id}` : "Tracking";
      const boxCoords = d.box ? `[${d.box.join(", ")}]` : "";

      let tagBadges = "";
      if (d.tags && d.tags.length > 0) {
        tagBadges = d.tags.map(t => `<span class="event-pill pill-triple_riding" style="font-size:0.65rem;padding:0.15rem 0.35rem;">${t}</span>`).join(" ");
      }

      card.innerHTML = `
        <div class="detection-card-top">
          <span class="detection-class-badge badge-${cname}">
            <span>${icon}</span>
            <span>${cname.toUpperCase()}</span>
          </span>
          <span class="detection-track-id">${trackId}</span>
        </div>

        <div class="conf-meter-bar">
          <div class="conf-bar-fill" style="width: ${confVal}%;"></div>
        </div>

        <div class="detection-card-bottom">
          <span style="color: var(--neon-cyan); font-weight: 600;">Conf: ${confPct}</span>
          <span style="color: var(--text-dim); font-size: 0.68rem;">${boxCoords}</span>
        </div>
        ${tagBadges ? `<div style="margin-top: 2px;">${tagBadges}</div>` : ''}
      `;

      liveDetectionsContainer.appendChild(card);
    });
  }

  // -------------------------------------------------------------
  // 8. Fetch Historical Events & Offline Stats
  // -------------------------------------------------------------
  async function fetchEvents() {
    try {
      const res = await fetch("/events");
      const data = await res.json();

      const events = data.events || [];
      events.forEach(ev => {
        const evKey = `${ev.timestamp}_${ev.event}_${ev.vehicle_id}`;
        if (!seenEventKeys.has(evKey)) {
          seenEventKeys.add(evKey);
          allEvents.push(ev);
        }
      });

      const stats = data.statistics || {};
      statTriple.textContent = stats.triple_riding || 0;
      statWrongWay.textContent = stats.wrong_way_driving || 0;
      statStopped.textContent = stats.vehicle_stopped || 0;
      statHelmet.textContent = stats.helmet_violation || 0;

      if (!isStreaming) {
        eventTotalBadge.textContent = `${data.total_events || allEvents.length} Events`;
      }

      renderEventsTable();
    } catch (err) {
      console.error("Error fetching events:", err);
    }
  }

  // -------------------------------------------------------------
  // 9. Render Event Log Table
  // -------------------------------------------------------------
  function renderEventsTable() {
    const filter = eventFilterSelect.value;
    const filtered = filter === "ALL" 
      ? allEvents 
      : allEvents.filter(e => e.event === filter);

    eventsTableBody.innerHTML = "";

    if (filtered.length === 0) {
      eventsTableBody.innerHTML = `
        <tr class="empty-row">
          <td colspan="6">No events matching "${filter}"</td>
        </tr>
      `;
      return;
    }

    filtered.slice(0, 100).forEach(ev => {
      const tr = document.createElement("tr");

      let timeStr = ev.timestamp || "N/A";
      if (timeStr.includes("T")) {
        timeStr = timeStr.split("T")[1];
      }

      let details = [];
      if (ev.person_count) details.push(`Riders: ${ev.person_count}`);
      if (ev.movement_direction) details.push(`Heading: ${ev.movement_direction}`);
      if (ev.stopped_duration_sec) details.push(`Stopped: ${ev.stopped_duration_sec}s`);
      if (ev.details && typeof ev.details === 'object') {
        for (const [k, v] of Object.entries(ev.details)) {
          details.push(`${k}: ${v}`);
        }
      }
      const detailsStr = details.join(" | ") || "Traffic Violation / Detection";

      const pillClass = `pill-${ev.event || 'object_detected'}`;
      const eventLabel = (ev.event || "DETECTED").replace(/_/g, " ");

      const confDisplay = ev.confidence !== undefined 
        ? `${Math.round(ev.confidence * 100)}%` 
        : "90%";

      tr.innerHTML = `
        <td style="font-family: var(--font-mono); font-size: 0.75rem;">${timeStr}</td>
        <td><code>${ev.camera_id || 'cam_01'}</code></td>
        <td><span class="event-pill ${pillClass}">${eventLabel}</span></td>
        <td><strong>#${ev.vehicle_id !== undefined ? ev.vehicle_id : 'N/A'}</strong></td>
        <td><span style="color: var(--neon-cyan);">${confDisplay}</span></td>
        <td style="color: var(--text-muted); font-size: 0.78rem;">${detailsStr}</td>
      `;
      eventsTableBody.appendChild(tr);
    });
  }

  eventFilterSelect.addEventListener("change", renderEventsTable);

  // -------------------------------------------------------------
  // 10. Batch Offline Video Processing
  // -------------------------------------------------------------
  btnRunProcess.addEventListener("click", async () => {
    const selectedVideo = videoSelect.value || "input.mp4";
    btnRunProcess.disabled = true;
    btnRunProcess.innerHTML = `
      <svg class="spinner" viewBox="0 0 50 50" style="width:16px;height:16px;animation:spin 1s linear infinite;">
        <circle cx="25" cy="25" r="20" fill="none" stroke="currentColor" stroke-width="5" stroke-dasharray="31.4 31.4"></circle>
      </svg>
      Processing Batch...
    `;

    try {
      const res = await fetch(`/process?video=${encodeURIComponent(selectedVideo)}`, { method: "POST" });
      const data = await res.json();

      if (!res.ok || data.status === "error") {
        showAlert("Batch Processing Error", data.message || "Failed to process video.", "error");
      } else {
        showAlert("Batch Processing Started", `AI Video processing running for ${selectedVideo}.`, "success");
      }

      await fetchEvents();
    } catch (err) {
      showAlert("Execution Error", err.message || "Could not connect to backend server.", "error");
    } finally {
      btnRunProcess.disabled = false;
      btnRunProcess.innerHTML = `
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
          <polyline points="14 2 14 8 20 8"></polyline>
          <line x1="12" y1="18" x2="12" y2="12"></line>
          <line x1="9" y1="15" x2="15" y2="15"></line>
        </svg>
        Batch Export MP4
      `;
    }
  });

  // Refresh Button Listener
  btnRefresh.addEventListener("click", () => {
    fetchEvents();
    loadVideosList();
  });

  // Initial Initialization
  loadVideosList();
  fetchEvents();

  // Periodic event refresh
  setInterval(fetchEvents, 5000);
});
