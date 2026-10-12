/*
  [목적] 화면 전환, 오디오 입력(업로드·녹음), 서버 분석 요청, 체크리스트 채점,
  결과 표시, 신고용 정보 생성까지 앱의 모든 동작을 담당한다.

  [프레임워크(React 등)를 쓰지 않은 이유] 화면이 6개인 단계별 마법사 수준이라
  순수 JavaScript로 충분하다. 프레임워크를 쓰면 npm·빌드도구·컴포넌트 개념을
  함께 익혀야 해서, 웹 입문 단계에서는 배보다 배꼽이 커진다.
*/

// ───────────────────────────────────────────────────────────────
// 1. 앱 상태
// [목적] 여러 단계에 걸쳐 유지돼야 하는 값을 한 곳에 모아둔다.
//        (분석은 2단계에서 시작하지만 결과는 3단계와 6단계에서 쓰인다)
// ───────────────────────────────────────────────────────────────
const state = {
  audioBlob: null,      // 업로드했거나 녹음한 오디오
  audioName: "",
  analysis: null,       // 서버 분석 결과 (도착 전에는 null)
  analysisError: null,
  checked: new Set(),   // 체크된 정황 항목 id
  damaged: null,        // true = 피해 발생, false = 없음
  startedAt: null,
};

// ───────────────────────────────────────────────────────────────
// 2. 정황 체크리스트
//
// [가중치 합산 대신 '결정적 신호'를 따로 둔 이유]
// 항목마다 점수를 매겨 더하는 방식은 숫자가 임의적이고 설명하기 어렵다.
// 실제로 "안전계좌로 이체하라"는 요구는 그 자체만으로 사기가 확실하다
// (수사기관·금융기관에 '안전계좌'라는 제도는 존재하지 않는다).
// 그래서 critical 항목은 하나만 체크돼도 최고 위험으로 올리고,
// 나머지는 개수로 판단한다. 규칙이 단순해서 사용자에게 설명하기도 쉽다.
// ───────────────────────────────────────────────────────────────
const CHECKLIST = [
  { id: "safe_account", critical: true,  label: "‘안전계좌’ 또는 ‘보호계좌’로 돈을 옮기라고 했다" },
  { id: "remote_app",   critical: true,  label: "특정 앱(원격제어·보안앱 등) 설치를 유도했다" },
  { id: "secret",       critical: true,  label: "가족·은행 직원에게 말하지 말라고 했다" },
  { id: "impersonate",  critical: false, label: "검찰·경찰·금감원·은행 직원이라고 밝혔다" },
  { id: "credentials",  critical: false, label: "계좌번호·비밀번호·OTP·주민번호를 요구했다" },
  { id: "cash",         critical: false, label: "현금 인출이나 직접 전달을 요구했다" },
  { id: "urgency",      critical: false, label: "지금 당장 하지 않으면 큰일 난다고 재촉했다" },
  { id: "unknown_num",  critical: false, label: "모르는 번호이거나 국제전화(+로 시작)였다" },
];

// 신고처 연락처
// TODO: 화면에 넣기 전에 각 기관 공식 안내에서 번호를 한 번 더 확인할 것.
const CONTACTS = [
  { who: "경찰청", num: "112", desc: "보이스피싱 신고·수사 의뢰 (가장 먼저)" },
  { who: "금융감독원", num: "1332", desc: "금융 피해 상담, 지급정지 안내" },
  { who: "한국인터넷진흥원", num: "118", desc: "개인정보 유출·악성앱 상담" },
];

const STEPS = ["step-input", "step-analyzing", "step-result", "step-damage", "step-report", "step-summary"];

// ───────────────────────────────────────────────────────────────
// 3. 화면 전환
// ───────────────────────────────────────────────────────────────
function showStep(id) {
  document.querySelectorAll(".step").forEach((el) => el.classList.remove("active"));
  document.getElementById(id).classList.add("active");
  renderProgress(STEPS.indexOf(id));
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function renderProgress(activeIdx) {
  const bar = document.getElementById("progress");
  bar.innerHTML = STEPS
    .map((_, i) => `<span class="${i <= activeIdx ? "done" : ""}"></span>`)
    .join("");
}

const $ = (id) => document.getElementById(id);

// ───────────────────────────────────────────────────────────────
// 4. 오디오 입력 — 파일 업로드
// ───────────────────────────────────────────────────────────────
$("btn-pick-file").addEventListener("click", () => $("file-input").click());

$("file-input").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  state.audioBlob = file;
  state.audioName = file.name;
  $("picked-name").textContent = `선택됨: ${file.name}`;
  startAnalysis();
});

// ───────────────────────────────────────────────────────────────
// 5. 오디오 입력 — 브라우저 녹음
//
// [MediaRecorder를 쓰는 이유] 브라우저에 내장된 표준 기능이라 추가 라이브러리가
// 필요 없다. 다만 마이크 접근은 보안상 HTTPS 또는 localhost에서만 허용된다.
// SSH 터널로 localhost:포트 에 접속하면 이 조건을 자동으로 만족한다.
//
// [webm으로 녹음돼도 되는 이유] 브라우저는 보통 webm/opus로 녹음하는데,
// 백엔드 파이프라인이 av 패키지로 opus를 디코딩할 수 있어 그대로 보내면 된다.
// ───────────────────────────────────────────────────────────────
let mediaRecorder = null;
let recordedChunks = [];

$("btn-record").addEventListener("click", async () => {
  const btn = $("btn-record");
  const status = $("record-status");

  // 녹음 중이면 중지
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordedChunks = [];
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) recordedChunks.push(e.data);
    };

    mediaRecorder.onstop = () => {
      // 마이크 사용 표시(브라우저 탭의 빨간 점)를 확실히 끄기 위해 트랙을 직접 정지한다.
      stream.getTracks().forEach((t) => t.stop());

      const blob = new Blob(recordedChunks, { type: "audio/webm" });
      state.audioBlob = blob;
      state.audioName = "recording.webm";
      btn.textContent = "녹음 시작";
      btn.classList.remove("btn-danger");
      status.textContent = `녹음 완료 (${(blob.size / 1024).toFixed(0)} KB)`;
      startAnalysis();
    };

    mediaRecorder.start();
    btn.textContent = "■ 녹음 중지";
    btn.classList.add("btn-danger");
    status.textContent = "녹음 중…";
  } catch (err) {
    status.textContent = "마이크를 사용할 수 없습니다: " + err.message;
  }
});

// ───────────────────────────────────────────────────────────────
// 6. 서버에 분석 요청
//
// [결과를 기다리지 않고 바로 2단계로 넘기는 이유]
// 분석에 수 초가 걸리는데 그동안 사용자를 빈 화면에서 기다리게 하면 낭비다.
// 대기 시간 동안 정황 체크리스트를 작성하게 하면 (1) 시간이 유용해지고
// (2) 모델 점수를 보기 전에 답하므로 결과에 끌려가는 편향을 막을 수 있다.
// ───────────────────────────────────────────────────────────────
// [목적] 이전 미리듣기의 Object URL을 남겨두지 않는다 (브라우저 메모리 누수 방지).
let audioPreviewUrl = null;

function showAudioPreview(blob, name) {
  if (audioPreviewUrl) URL.revokeObjectURL(audioPreviewUrl);
  audioPreviewUrl = URL.createObjectURL(blob);
  $("audio-preview").src = audioPreviewUrl;
  $("audio-preview-name").textContent = `${name} · ${(blob.size / 1024).toFixed(0)} KB`;
  $("audio-preview-card").style.display = "block";
}

function startAnalysis() {
  state.startedAt = new Date();
  state.analysis = null;
  state.analysisError = null;
  showStep("step-analyzing");
  setAnalysisStatus("분석 중입니다…", "모델이 오디오를 처리하고 있습니다", false);
  showAudioPreview(state.audioBlob, state.audioName);

  const form = new FormData();
  form.append("audio", state.audioBlob, state.audioName);

  fetch("/analyze", { method: "POST", body: form })
    .then(async (res) => {
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.detail || data.error || "분석 실패");
      return data;
    })
    .then((data) => {
      state.analysis = data;
      const src = data.mode === "dummy" ? " (모델 미연결 · 예시 데이터)" : "";
      setAnalysisStatus("분석 완료" + src, `처리 시간 ${data.elapsed_sec}초`, true);
    })
    .catch((err) => {
      // 분석이 실패해도 앱을 멈추지 않는다. 정황 체크만으로도 유의미한 안내가 가능하다.
      state.analysisError = err.message;
      setAnalysisStatus("분석에 실패했습니다", "정황 확인만으로 계속 진행할 수 있습니다", true);
    });
}

function setAnalysisStatus(title, detail, done) {
  $("analysis-title").textContent = title;
  $("analysis-detail").textContent = detail;
  $("spinner").classList.toggle("done", done);
}

// 긴급 경로: 오디오 없이 정황 확인만
$("btn-skip-to-check").addEventListener("click", () => {
  state.startedAt = new Date();
  showStep("step-analyzing");
  setAnalysisStatus("오디오 분석 건너뜀", "정황 확인만으로 진행합니다", true);
  $("audio-preview-card").style.display = "none";
});

// ───────────────────────────────────────────────────────────────
// 7. 체크리스트 렌더링 + 채점
// ───────────────────────────────────────────────────────────────
function renderChecklist() {
  $("checklist").innerHTML = CHECKLIST.map((item) => `
    <label class="check-item" data-id="${item.id}">
      <input type="checkbox" value="${item.id}">
      <span class="label">${item.label}${item.critical ? '<span class="tag">결정적</span>' : ""}</span>
    </label>
  `).join("");

  $("checklist").querySelectorAll("input").forEach((input) => {
    input.addEventListener("change", (e) => {
      const id = e.target.value;
      const row = e.target.closest(".check-item");
      if (e.target.checked) { state.checked.add(id); row.classList.add("checked"); }
      else { state.checked.delete(id); row.classList.remove("checked"); }
    });
  });
}
renderChecklist();

function scoreChecklist() {
  const hits = CHECKLIST.filter((c) => state.checked.has(c.id));
  const criticalHits = hits.filter((c) => c.critical);

  return { hits, criticalHits };
}

// 정황 점수와 모델 결과를 합쳐 최종 위험 등급을 정한다.
//
// [모델을 등급 산정에 넣는 이유] 정황만으로 등급을 매기면, 체크 항목이 하나도
// 없을 때 모델이 74%를 찍어도 헤드라인이 초록색으로 나와 한눈에 "안전"으로
// 읽힌다. 모델이 참고용이라도 높은 점수를 무시하고 안심시키는 화면을 만들면
// 안 되므로, 그 경우 등급을 '주의'로 올린다.
//
// [그래도 정황이 우선인 이유] 반대로 모델 점수가 낮다고 해서 정황 기반 등급을
// 낮추지는 않는다. 모델의 음성 EER이 약 22%라 낮은 점수는 신뢰할 수 없다.
// 즉 모델은 등급을 올릴 수는 있어도 내릴 수는 없다.
function computeVerdict() {
  const score = scoreChecklist();
  const a = state.analysis;
  const modelHigh = !!(a && a.reliable.voice && a.fake.voice >= 0.5);

  let level, text;
  if (score.criticalHits.length > 0) {
    level = "critical";
    text = "매우 높음 — 보이스피싱이 거의 확실합니다";
  } else if (score.hits.length >= 3) {
    level = "high";
    text = "높음 — 보이스피싱 가능성이 큽니다";
  } else if (score.hits.length >= 1) {
    level = "warn";
    text = "주의 — 의심스러운 정황이 있습니다";
  } else if (modelHigh) {
    level = "warn";
    text = "주의 — 모델이 AI 생성 음성일 가능성을 높게 봤습니다";
  } else {
    level = "low";
    text = "확인된 위험 신호 없음";
  }
  return { ...score, level, text, modelHigh };
}

// ───────────────────────────────────────────────────────────────
// 8. 결과 화면
// ───────────────────────────────────────────────────────────────
$("btn-to-result").addEventListener("click", () => {
  renderResult();
  showStep("step-result");
});

function renderResult() {
  const score = computeVerdict();

  // ── 종합 판단: 정황이 1차, 모델은 보조 ───────────────────────
  $("verdict").className = "verdict " + score.level;
  $("verdict").textContent = score.text;
  $("verdict-detail").innerHTML = buildVerdictDetail(score);

  // ── 정황 상세 ────────────────────────────────────────────
  $("checklist-result").innerHTML = score.hits.length
    ? `<p>체크된 항목 <strong>${score.hits.length}개</strong>${score.criticalHits.length ? ` (결정적 신호 ${score.criticalHits.length}개 포함)` : ""}</p>
       <ul class="hit-list">${score.hits.map((h) => `<li>${h.label}</li>`).join("")}</ul>`
    : `<p class="muted">체크된 항목이 없습니다. 다만 체크리스트에 없는 수법도 많으니 방심하지 마세요.</p>`;

  // ── 모델 결과 ────────────────────────────────────────────
  const a = state.analysis;
  // 분석 결과가 없으면 경고문 박스 자체를 숨긴다.
  // (textContent만 비우면 빈 노란 상자가 그대로 남는다)
  $("disclaimer").style.display = a ? "block" : "none";
  if (!a) {
    $("model-result").innerHTML = `<p class="muted">${state.analysisError ? "분석 실패: " + state.analysisError : "오디오 분석을 하지 않았습니다."}</p>`;
    return;
  }

  const parts = [];
  parts.push(bar("음성 검출", a.presence.voice, "존재 확률"));
  if (a.reliable.voice) {
    parts.push(bar("음성 AI 생성 가능성", a.fake.voice, "모델 추정"));
    // [실험용 비교치] 분리(separated)가 실행된 파일에서만 채워진다. 최종
    // 판정(위 막대)엔 영향을 주지 않고, 곁다리 실험 모델의 참고 점수만 보여준다.
    if (a.fake.voice_side_probe != null) {
      parts.push(bar("음성 AI 생성 가능성 (실험용 side 모델)", a.fake.voice_side_probe, "비교 참고용 · 최종 판정 아님"));
    }
  } else {
    parts.push(`<p class="bar-note">음성이 거의 검출되지 않아 음성 판정은 참고할 수 없습니다.</p>`);
  }
  parts.push(bar("음악 검출", a.presence.music, "존재 확률"));
  if (a.reliable.music) {
    parts.push(bar("음악 AI 생성 가능성", a.fake.music, "모델 추정"));
  }
  const z = a.pipeline.zones || {};
  const zoneTxt = z.voice ? ` · 판정 구간 음성 ${z.voice} / 음악 ${z.music}` : "";
  parts.push(`<p class="bar-note">처리 경로: ${a.pipeline.route} · 음원 분리 ${a.pipeline.separated ? "실행" : "생략"}${zoneTxt} · 총 ${a.elapsed_sec}초</p>`);

  // 단계별 처리 시간 — 파이프라인이 실제로 어떤 모델들을 거쳤는지 보여준다.
  const stages = a.pipeline.stages || [];
  if (stages.length) {
    const max = Math.max(...stages.map((s) => s.sec));
    parts.push(`<div class="stages"><div class="stages-title">처리 단계</div>` +
      stages.map((s) => `
        <div class="stage">
          <span class="stage-name">${s.name}</span>
          <span class="stage-bar"><i style="width:${Math.max(2, (s.sec / max) * 100)}%"></i></span>
          <span class="stage-sec">${s.sec.toFixed(2)}s</span>
        </div>`).join("") + `</div>`);
  }

  if (a.mode === "dummy") {
    parts.push(`<p class="bar-note"><strong>※ 모델 미연결 상태의 예시 데이터입니다.</strong></p>`);
  }

  $("model-result").innerHTML = parts.join("");
  $("disclaimer").textContent = a.disclaimer;
}

// 확률 막대 하나를 그린다.
function bar(label, value, note) {
  const pct = (value * 100).toFixed(1);
  const cls = value >= 0.6 ? "fill-high" : value >= 0.3 ? "fill-mid" : "fill-low";
  return `
    <div class="bar-row">
      <div class="bar-label"><span>${label} <span class="muted small">${note}</span></span><span class="val">${pct}%</span></div>
      <div class="bar"><div class="${cls}" style="width:${pct}%"></div></div>
    </div>`;
}

// 정황과 모델 결과를 함께 해석한 문장을 만든다.
// [낮은 확률에 '안전'이라고 쓰지 않는 이유] 모델의 음성 EER이 약 22%다.
// 다섯 건 중 한 건은 틀리는 도구의 낮은 점수를 근거로 안심시키면,
// 실제 피해자가 경계를 푸는 최악의 결과로 이어질 수 있다.
function buildVerdictDetail(score) {
  const modelHigh = score.modelHigh;

  if (score.level === "critical" || score.level === "high") {
    return `정황만으로도 위험이 충분히 확인됩니다. 모델 결과와 무관하게 <strong>즉시 통화를 중단하고 송금하지 마세요.</strong>`;
  }
  if (score.level === "warn") {
    if (score.hits.length === 0) {
      // 정황은 없는데 모델만 높게 본 경우
      return `체크된 정황은 없지만 모델이 AI 생성 가능성을 높게 봤습니다. 체크리스트에 없는 수법일 수 있으니 <strong>금전 요구가 있었다면 반드시 해당 기관 대표번호로 직접 확인하세요.</strong>`;
    }
    return modelHigh
      ? `의심 정황이 있고 모델도 AI 생성 가능성을 높게 봤습니다. <strong>통화를 중단하고 해당 기관 대표번호로 직접 확인하세요.</strong>`
      : `의심 정황이 있습니다. 모델 점수는 낮지만 <strong>모델은 참고용일 뿐이므로 이를 근거로 안심하지 마세요.</strong> 상대가 요구한 내용을 해당 기관 대표번호로 직접 확인하세요.`;
  }
  // 정황도 없고 모델도 높지 않은 경우
  return `현재까지 확인된 위험 신호는 없습니다. 다만 이 결과가 <strong>안전을 보장하지는 않습니다.</strong> 조금이라도 금전이나 개인정보 요구가 있었다면 해당 기관 대표번호로 직접 확인하세요.`;
}

// ───────────────────────────────────────────────────────────────
// 9. 피해 여부 → 신고 안내
// ───────────────────────────────────────────────────────────────
$("btn-to-damage").addEventListener("click", () => showStep("step-damage"));

$("btn-damage-yes").addEventListener("click", () => { state.damaged = true; renderReportGuide(); showStep("step-report"); });
$("btn-damage-no").addEventListener("click", () => { state.damaged = false; renderReportGuide(); showStep("step-report"); });

function renderReportGuide() {
  const contacts = CONTACTS.map((c) => `
    <div class="contact">
      <div><div class="who">${c.who}</div><div class="desc">${c.desc}</div></div>
      <div class="num">${c.num}</div>
    </div>`).join("");

  const urgent = `
    <div class="card urgent">
      <strong>지금 바로 해야 할 일 (순서대로)</strong>
      <ol class="step-list">
        <li><strong>송금한 은행 고객센터에 전화해 지급정지를 요청하세요.</strong> 시간이 가장 중요합니다 — 상대가 돈을 인출하기 전이면 막을 수 있습니다.</li>
        <li><strong>112로 신고</strong>하고 사건접수번호를 받으세요.</li>
        <li>은행에 <strong>피해구제 신청서</strong>를 제출하세요 (지급정지 후 3영업일 이내).</li>
        <li>악성 앱을 설치했다면 휴대폰을 <strong>비행기 모드로 전환</strong>하고, 다른 기기로 연락하세요.</li>
      </ol>
    </div>`;

  const prevent = `
    <div class="card">
      <h2>5. 지금 하실 일</h2>
      <ol class="step-list">
        <li>해당 번호를 <strong>차단</strong>하세요.</li>
        <li>상대가 기관을 사칭했다면, 그 기관 <strong>대표번호로 직접 전화</strong>해 사실 여부를 확인하세요. 상대가 알려준 번호는 절대 쓰지 마세요.</li>
        <li>아직 피해가 없어도 <strong>신고하면 다른 피해를 막는 데 도움</strong>이 됩니다.</li>
        <li>설치한 앱이 있다면 즉시 삭제하고, 금융 비밀번호를 변경하세요.</li>
      </ol>
    </div>`;

  $("report-guide").innerHTML =
    (state.damaged ? urgent : prevent) +
    `<div class="card"><h3>신고·상담 연락처</h3>${contacts}</div>`;
}

// ───────────────────────────────────────────────────────────────
// 10. 신고용 정보 정리
//
// [자동 신고를 하지 않는 이유] 모델이 틀릴 수 있어 오탐 시 허위신고가 되고,
// 실제 신고는 본인 진술이 필요하다. 정보를 정리해 주는 데까지가 이 도구의
// 적절한 범위다.
// ───────────────────────────────────────────────────────────────
$("btn-to-summary").addEventListener("click", () => {
  $("summary-text").value = buildSummary();
  showStep("step-summary");
});

function buildSummary() {
  const score = computeVerdict();
  const a = state.analysis;
  const when = (state.startedAt || new Date()).toLocaleString("ko-KR");
  const L = [];

  L.push("[보이스피싱 의심 신고용 정리]");
  L.push(`작성 일시: ${when}`);
  L.push("");
  L.push("■ 종합 판단");
  L.push(`  정황 위험도: ${score.text}`);
  L.push(`  피해 발생: ${state.damaged === null ? "미확인" : state.damaged ? "있음 (송금 또는 정보 제공)" : "없음"}`);
  L.push("");

  L.push("■ 확인된 의심 정황");
  if (score.hits.length) {
    score.hits.forEach((h, i) => L.push(`  ${i + 1}. ${h.label}${h.critical ? "  [결정적 신호]" : ""}`));
  } else {
    L.push("  (체크된 항목 없음)");
  }
  L.push("");

  L.push("■ 오디오 AI 생성 분석 (참고용)");
  if (a) {
    L.push(`  파일: ${a.filename || state.audioName || "-"}`);
    L.push(`  음성 검출 확률: ${(a.presence.voice * 100).toFixed(1)}%`);
    L.push(`  음성 AI 생성 가능성: ${a.reliable.voice ? (a.fake.voice * 100).toFixed(1) + "%" : "판정 불가 (음성 미검출)"}`);
    L.push(`  음악 검출 확률: ${(a.presence.music * 100).toFixed(1)}%`);
    L.push(`  음악 AI 생성 가능성: ${a.reliable.music ? (a.fake.music * 100).toFixed(1) + "%" : "판정 불가 (음악 미검출)"}`);
    L.push(`  처리 경로: ${a.pipeline.route} / 음원 분리 ${a.pipeline.separated ? "실행" : "생략"}`);
    if (a.mode === "dummy") L.push("  ※ 모델 미연결 상태의 예시 데이터임");
    L.push("");
    L.push("  ※ 이 분석은 보조 지표입니다. 음성 판정의 동일오류율(EER)이 약 22%로,");
    L.push("     다섯 건 중 한 건 정도는 틀릴 수 있어 단독 증거로 쓸 수 없습니다.");
  } else {
    L.push("  (오디오 분석을 수행하지 않았거나 실패함)");
  }
  L.push("");

  L.push("■ 추가로 기재할 내용 (직접 작성하세요)");
  L.push("  - 상대방 전화번호:");
  L.push("  - 통화 일시 및 시간:");
  L.push("  - 상대가 사칭한 기관/직책:");
  L.push("  - 송금한 계좌번호 및 금액:");
  L.push("  - 통화 내용 요약:");
  L.push("");
  L.push("■ 신고처");
  CONTACTS.forEach((c) => L.push(`  ${c.who} ${c.num} — ${c.desc}`));

  return L.join("\n");
}

$("btn-copy").addEventListener("click", async () => {
  const text = $("summary-text").value;
  try {
    await navigator.clipboard.writeText(text);
    $("copy-status").textContent = "복사했습니다.";
  } catch {
    // clipboard API는 보안 컨텍스트(HTTPS/localhost)에서만 동작한다.
    // 실패하면 사용자가 직접 선택해 복사할 수 있도록 안내한다.
    $("summary-text").select();
    $("copy-status").textContent = "자동 복사에 실패했습니다. Cmd+C 로 복사하세요.";
  }
});

$("btn-download").addEventListener("click", () => {
  const blob = new Blob([$("summary-text").value], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `보이스피싱_신고정리_${new Date().toISOString().slice(0, 10)}.txt`;
  a.click();
  URL.revokeObjectURL(url);
});

$("btn-restart").addEventListener("click", () => window.location.reload());

// 첫 화면 표시
showStep("step-input");
