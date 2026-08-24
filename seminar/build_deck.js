const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
p.author = "RetroInfer Seminar";

// ---- palette ----
const INK   = "0C1B33"; // deep navy (dark bg)
const INK2  = "13284F";
const NAVY  = "1E3A8A";
const TEAL  = "14B8A6";
const CYAN  = "38BDF8";
const AMBER = "F59E0B";
const WHITE = "FFFFFF";
const CARD  = "F1F5F9";
const CARD2 = "E8EEF5";
const TEXT  = "0F172A";
const MUTED = "64748B";
const LINE  = "CBD5E1";

const F = "Malgun Gothic";     // Korean-capable body
const FB = "Malgun Gothic";    // headers (bold applied)
const W = 13.33, H = 7.5;

// ---- helpers ----
function pageNum(s, n, dark) {
  s.addText(String(n).padStart(2, "0"), {
    x: W - 0.9, y: H - 0.5, w: 0.5, h: 0.3, align: "right",
    fontFace: F, fontSize: 10, color: dark ? "6B7A99" : MUTED, margin: 0,
  });
}
function footer(s, dark) {
  s.addText("RetroInfer · 기술 심화 세미나", {
    x: 0.5, y: H - 0.5, w: 6, h: 0.3, align: "left",
    fontFace: F, fontSize: 10, color: dark ? "6B7A99" : MUTED, margin: 0,
  });
}
function title(s, t, sub) {
  s.addText(t, { x: 0.6, y: 0.45, w: W - 1.2, h: 0.7, fontFace: FB, fontSize: 30, bold: true, color: TEXT, margin: 0 });
  if (sub) s.addText(sub, { x: 0.62, y: 1.16, w: W - 1.2, h: 0.4, fontFace: F, fontSize: 14, color: TEAL, bold: true, margin: 0 });
}
function chip(s, x, y, label, fill) {
  s.addShape(p.ShapeType.roundRect, { x, y, w: 0.42, h: 0.42, rectRadius: 0.09, fill: { color: fill } });
  s.addText(label, { x, y, w: 0.42, h: 0.42, align: "center", valign: "middle", fontFace: FB, fontSize: 15, bold: true, color: WHITE, margin: 0 });
}

// =========================================================
// Slide 1 — Title (dark)
// =========================================================
let s = p.addSlide();
s.background = { color: INK };
// subtle secondary block
s.addShape(p.ShapeType.rect, { x: 0, y: 0, w: W, h: 0.14, fill: { color: TEAL } });
// vector-cluster motif (right)
const gx = 8.4, gy = 1.5, step = 0.62, cols = 7, rows = 7;
const teal = new Set(["1,2","2,2","2,3","3,2","4,4","5,4","4,5"]);
const qcell = "3,3";
for (let r = 0; r < rows; r++) {
  for (let c = 0; c < cols; c++) {
    const key = c + "," + r;
    const cx = gx + c * step, cy = gy + r * step;
    let col = "22345F", d = 0.16;
    if (teal.has(key)) { col = TEAL; d = 0.2; }
    if (key === qcell) { col = AMBER; d = 0.26; }
    s.addShape(p.ShapeType.ellipse, { x: cx, y: cy, w: d, h: d, fill: { color: col } });
  }
}
// connecting lines from query to teal cluster
[["1,2"],["2,3"],["4,4"]].forEach(([k]) => {
  const [c, r] = k.split(",").map(Number);
  const qc = 3, qr = 3;
  s.addShape(p.ShapeType.line, {
    x: gx + qc*step + 0.13, y: gy + qr*step + 0.13,
    w: (c-qc)*step, h: (r-qr)*step,
    line: { color: CYAN, width: 1, transparency: 40 },
  });
});
s.addText("RetroInfer", { x: 0.7, y: 2.0, w: 7.4, h: 1.1, fontFace: FB, fontSize: 60, bold: true, color: WHITE, margin: 0 });
s.addText("KV 캐시를 벡터 저장 시스템으로 재해석한\n확장 가능한 긴 컨텍스트 LLM 추론", {
  x: 0.72, y: 3.15, w: 7.3, h: 1.2, fontFace: F, fontSize: 20, color: "CADCFC", lineSpacingMultiple: 1.15, margin: 0 });
s.addText([
  { text: "Attention-aWare VEctor index (wave index)", options: { color: TEAL, bold: true } },
  { text: "   +   Wave Buffer (GPU–CPU 협력)", options: { color: "8AA0C8" } },
], { x: 0.72, y: 4.45, w: 7.4, h: 0.4, fontFace: F, fontSize: 14, margin: 0 });
s.addText("기술 심화 세미나  ·  VLDB 2026 / NeurIPS 2025  ·  arXiv:2505.02922", {
  x: 0.72, y: 6.5, w: 9, h: 0.4, fontFace: F, fontSize: 12, color: "6B7A99", margin: 0 });

// =========================================================
// Slide 2 — Problem (light)
// =========================================================
s = p.addSlide();
s.background = { color: WHITE };
title(s, "문제: 긴 컨텍스트 디코딩의 병목", "매 토큰 생성마다 전체 KV 캐시와 어텐션 → 계산·메모리가 컨텍스트 길이에 선형 증가");
// left bullets
const probs = [
  ["전량 어텐션", "자기회귀 디코딩은 매 스텝 모든 과거 토큰의 K·V와 내적 → O(N) 계산·메모리 접근"],
  ["메모리 압박", "120K~1M 토큰에서 KV 캐시가 수십 GB → GPU 메모리를 압도"],
  ["처리량 붕괴", "FlashAttention도 전체 KV 접근은 불가피 → 배치·처리량이 급격히 하락"],
];
let y = 1.9;
probs.forEach((b, i) => {
  s.addShape(p.ShapeType.roundRect, { x: 0.6, y, w: 6.0, h: 1.35, rectRadius: 0.08, fill: { color: CARD }, line: { color: LINE, width: 1 } });
  chip(s, 0.85, y + 0.24, String(i + 1), NAVY);
  s.addText(b[0], { x: 1.42, y: y + 0.16, w: 5.0, h: 0.4, fontFace: FB, fontSize: 16, bold: true, color: TEXT, margin: 0 });
  s.addText(b[1], { x: 1.42, y: y + 0.58, w: 5.0, h: 0.7, fontFace: F, fontSize: 12.5, color: "334155", lineSpacingMultiple: 1.05, margin: 0 });
  y += 1.5;
});
// right: full-attention schematic
const rx = 7.1, rw = 5.6;
s.addShape(p.ShapeType.roundRect, { x: rx, y: 1.9, w: rw, h: 4.55, rectRadius: 0.1, fill: { color: INK } });
s.addText("Full Attention", { x: rx, y: 2.05, w: rw, h: 0.35, align: "center", fontFace: FB, fontSize: 13, bold: true, color: CYAN, margin: 0 });
// query node
const qx = rx + 0.7, qy = 4.2;
// key tokens on right
const kn = 9, kx = rx + rw - 0.95;
for (let i = 0; i < kn; i++) {
  const ky = 2.6 + i * 0.42;
  s.addShape(p.ShapeType.line, { x: qx + 0.28, y: qy + 0.14, w: (kx) - (qx + 0.28), h: ky + 0.11 - (qy + 0.14), line: { color: "3B5687", width: 0.75 } });
}
for (let i = 0; i < kn; i++) {
  const ky = 2.6 + i * 0.42;
  s.addShape(p.ShapeType.ellipse, { x: kx, y: ky, w: 0.22, h: 0.22, fill: { color: TEAL } });
}
s.addShape(p.ShapeType.ellipse, { x: qx, y: qy, w: 0.46, h: 0.46, fill: { color: AMBER } });
s.addText("query", { x: qx - 0.3, y: qy + 0.5, w: 1.05, h: 0.3, align: "center", fontFace: F, fontSize: 11, color: "CBD5E1", margin: 0 });
s.addText("모든 KV 토큰", { x: kx - 1.6, y: 2.55, w: 1.4, h: 0.3, align: "right", fontFace: F, fontSize: 11, color: "CBD5E1", margin: 0 });
s.addText("계산량 ∝ 컨텍스트 길이 N", { x: rx + 0.3, y: 5.95, w: rw - 0.6, h: 0.35, align: "center", fontFace: FB, fontSize: 12, italic: true, color: "94A3B8", margin: 0 });
footer(s); pageNum(s, 2);

// =========================================================
// Slide 3 — Key insight (dark highlight)
// =========================================================
s = p.addSlide();
s.background = { color: INK };
s.addText("핵심 통찰", { x: 0.6, y: 0.5, w: 6, h: 0.5, fontFace: FB, fontSize: 18, bold: true, color: TEAL, margin: 0 });
s.addText("어텐션은 희소하다 — KV 캐시를 검색 가능한 벡터 DB로", {
  x: 0.6, y: 1.05, w: W - 1.2, h: 1.2, fontFace: FB, fontSize: 33, bold: true, color: WHITE, lineSpacingMultiple: 1.05, margin: 0 });
const ins = [
  ["희소성(Sparsity)", "실제로 중요한 토큰은 소수뿐. 전량 어텐션은 대부분 불필요한 계산."],
  ["근사 최근접 검색", "query에 관련된 KV 찾기 = ANN 문제 → 벡터 검색 엔진의 전형적 작업."],
  ["정확도 보장", "3-zone 분해로 동적 희소성을 오차 경계 안에서 처리 → 정확도 손실 최소화."],
];
ins.forEach((b, i) => {
  const cx = 0.6 + i * 4.15;
  s.addShape(p.ShapeType.roundRect, { x: cx, y: 2.75, w: 3.85, h: 2.5, rectRadius: 0.1, fill: { color: INK2 }, line: { color: "27406E", width: 1 } });
  s.addShape(p.ShapeType.ellipse, { x: cx + 0.3, y: 3.05, w: 0.6, h: 0.6, fill: { color: TEAL } });
  s.addText(String(i + 1), { x: cx + 0.3, y: 3.05, w: 0.6, h: 0.6, align: "center", valign: "middle", fontFace: FB, fontSize: 20, bold: true, color: INK, margin: 0 });
  s.addText(b[0], { x: cx + 0.3, y: 3.85, w: 3.25, h: 0.5, fontFace: FB, fontSize: 16, bold: true, color: WHITE, margin: 0 });
  s.addText(b[1], { x: cx + 0.3, y: 4.35, w: 3.3, h: 1.0, fontFace: F, fontSize: 12.5, color: "AEBEDD", lineSpacingMultiple: 1.1, margin: 0 });
});
s.addText([
  { text: "결론:  ", options: { color: AMBER, bold: true } },
  { text: "KV 캐시를 “저장 버퍼”가 아니라 “벡터 저장 엔진”으로 다룬다.", options: { color: "CADCFC" } },
], { x: 0.6, y: 5.7, w: W - 1.2, h: 0.6, fontFace: FB, fontSize: 17, margin: 0 });
footer(s, true); pageNum(s, 3, true);

// =========================================================
// Slide 4 — Architecture (light)
// =========================================================
s = p.addSlide();
s.background = { color: WHITE };
title(s, "시스템 아키텍처", "Wave Index로 무엇을 검색할지 결정하고, Wave Buffer로 어디에 두고 어떻게 옮길지 조율");
// pipeline boxes
function box(x, yy, w, h, fill, lc, head, headColor, body, bodyColor) {
  s.addShape(p.ShapeType.roundRect, { x, y: yy, w, h, rectRadius: 0.09, fill: { color: fill }, line: { color: lc, width: 1 } });
  s.addText(head, { x: x + 0.2, y: yy + 0.16, w: w - 0.4, h: 0.4, fontFace: FB, fontSize: 15, bold: true, color: headColor, margin: 0 });
  if (body) s.addText(body, { x: x + 0.2, y: yy + 0.62, w: w - 0.4, h: h - 0.75, fontFace: F, fontSize: 11.5, color: bodyColor, lineSpacingMultiple: 1.08, margin: 0 });
}
function arrow(x, yy, w) {
  s.addShape(p.ShapeType.line, { x, y: yy, w, h: 0, line: { color: NAVY, width: 2, endArrowType: "triangle" } });
}
// Query
box(0.6, 2.2, 2.05, 1.15, INK, INK, "Query", CYAN, "디코딩 1스텝\n쿼리 벡터", "CBD5E1");
arrow(2.72, 2.78, 0.5);
// Wave Index
box(3.32, 1.95, 3.15, 3.6, "EAF6F4", TEAL, "① Wave Index (GPU)", "0F766E",
  "segmented k-means로\nKV를 클러스터링\n\n· centroid = 인덱스 엔트리\n· query·centroid 유사도\n· top-nprobe 클러스터 선택", "134E4A");
arrow(6.55, 3.75, 0.5);
// Wave Buffer
box(7.15, 1.95, 3.35, 3.6, "EEF2FF", NAVY, "② Wave Buffer", "1E3A8A",
  "GPU–CPU 협력 저장\n\n· 전체 KV는 CPU에 상주\n· 선택 클러스터만 GPU로 gather\n· LRU 블록 캐시\n· 연산·전송 overlap", "1E293B");
arrow(10.58, 3.75, 0.5);
// Attention out
box(11.18, 2.2, 1.55, 1.15, TEAL, TEAL, "Output", WHITE, "3-zone\n어텐션 결과", "ECFEFF");
// CPU store strip under wave buffer
s.addShape(p.ShapeType.roundRect, { x: 7.15, y: 5.75, w: 3.35, h: 0.62, rectRadius: 0.06, fill: { color: CARD2 }, line: { color: LINE, width: 1 } });
s.addText("CPU 대용량 KV 저장소 (수십~수백 GB)", { x: 7.15, y: 5.75, w: 3.35, h: 0.62, align: "center", valign: "middle", fontFace: F, fontSize: 11, color: "334155", margin: 0 });
s.addShape(p.ShapeType.line, { x: 8.82, y: 5.55, w: 0, h: 0.2, line: { color: NAVY, width: 1.5, endArrowType: "triangle", beginArrowType: "triangle" } });
footer(s); pageNum(s, 4);

// =========================================================
// Slide 5 — 3-Zone attention (light)
// =========================================================
s = p.addSlide();
s.background = { color: WHITE };
title(s, "정확도 보장 어텐션: 3개의 존(Zone)", "accuracy-bounded attention estimation — 동적 희소성을 오차 경계 안에서 처리");
const zones = [
  ["Steady", "8B5CF6", "고정 토큰 (어텐션 싱크·최근 토큰)", "항상 정확 어텐션", "steady_zone_keys / values"],
  ["Retrieval", TEAL, "query와 관련 높은 nprobe 클러스터", "검색 후 정확 어텐션", "nprobe = ⌈n_centroids × retrieval_budget⌉"],
  ["Estimation", AMBER, "나머지 클러스터", "centroid로 근사 추정", "es = ⌈n_centroids × estimation_budget⌉"],
];
zones.forEach((z, i) => {
  const cx = 0.6 + i * 4.15;
  s.addShape(p.ShapeType.roundRect, { x: cx, y: 1.95, w: 3.85, h: 3.9, rectRadius: 0.1, fill: { color: CARD }, line: { color: LINE, width: 1 } });
  s.addShape(p.ShapeType.roundRect, { x: cx, y: 1.95, w: 3.85, h: 0.72, rectRadius: 0.1, fill: { color: z[1] } });
  s.addShape(p.ShapeType.rect, { x: cx, y: 2.4, w: 3.85, h: 0.27, fill: { color: z[1] } });
  s.addText(z[0], { x: cx + 0.25, y: 1.95, w: 3.35, h: 0.72, valign: "middle", fontFace: FB, fontSize: 19, bold: true, color: WHITE, margin: 0 });
  s.addText("대상", { x: cx + 0.25, y: 2.85, w: 3.35, h: 0.3, fontFace: FB, fontSize: 11, bold: true, color: MUTED, margin: 0 });
  s.addText(z[2], { x: cx + 0.25, y: 3.12, w: 3.35, h: 0.7, fontFace: F, fontSize: 13, color: TEXT, lineSpacingMultiple: 1.05, margin: 0 });
  s.addText("계산", { x: cx + 0.25, y: 3.9, w: 3.35, h: 0.3, fontFace: FB, fontSize: 11, bold: true, color: MUTED, margin: 0 });
  s.addText(z[3], { x: cx + 0.25, y: 4.17, w: 3.35, h: 0.4, fontFace: FB, fontSize: 13.5, bold: true, color: z[1] === AMBER ? "B45309" : (z[1] === TEAL ? "0F766E" : "6D28D9"), margin: 0 });
  s.addShape(p.ShapeType.roundRect, { x: cx + 0.25, y: 4.75, w: 3.35, h: 0.85, rectRadius: 0.05, fill: { color: "0F172A" } });
  s.addText(z[4], { x: cx + 0.35, y: 4.75, w: 3.15, h: 0.85, valign: "middle", fontFace: "Consolas", fontSize: 10.5, color: "E2E8F0", margin: 0 });
});
s.addText("세 존의 결과를 online-softmax로 병합 → 전량 어텐션에 근접하되 계산은 극히 일부만", {
  x: 0.6, y: 6.1, w: W - 1.2, h: 0.4, align: "center", fontFace: FB, fontSize: 13.5, italic: true, color: "334155", margin: 0 });
footer(s); pageNum(s, 5);

// =========================================================
// Slide 6 — Wave Index (light)
// =========================================================
s = p.addSlide();
s.background = { color: WHITE };
title(s, "Wave Index: 어텐션 인식 벡터 인덱스", "segmented clustering으로 저오버헤드 인덱스 구축 (cache_hub/kmeans.py)");
// left explanation
const wi = [
  ["분할 클러스터링", "시퀀스를 세그먼트로 나눠 세그먼트별 k-means → 공간적 지역성 활용, 낮은 구축 비용"],
  ["centroid 색인", "각 클러스터의 대표 벡터 = 벡터 DB의 인덱스 엔트리 (self.centroids)"],
  ["검색", "batch_gemm_softmax로 Softmax(Q·Cᵀ) → top-k 클러스터 선택 (retrieval + estimation)"],
];
y = 2.0;
wi.forEach((b, i) => {
  chip(s, 0.6, y, String(i + 1), TEAL);
  s.addText(b[0], { x: 1.2, y: y - 0.04, w: 5.4, h: 0.4, fontFace: FB, fontSize: 15.5, bold: true, color: TEXT, margin: 0 });
  s.addText(b[1], { x: 1.2, y: y + 0.38, w: 5.5, h: 0.8, fontFace: F, fontSize: 12.5, color: "334155", lineSpacingMultiple: 1.08, margin: 0 });
  y += 1.42;
});
// right diagram: tokens -> clusters -> centroids -> topk
const dx = 7.2, dw = 5.5;
s.addShape(p.ShapeType.roundRect, { x: dx, y: 1.9, w: dw, h: 4.55, rectRadius: 0.1, fill: { color: CARD }, line: { color: LINE, width: 1 } });
// token dots
s.addText("KV 토큰", { x: dx + 0.25, y: 2.05, w: 1.5, h: 0.3, fontFace: FB, fontSize: 11, bold: true, color: MUTED, margin: 0 });
for (let i = 0; i < 24; i++) {
  const cxx = dx + 0.3 + (i % 6) * 0.2, cyy = 2.45 + Math.floor(i / 6) * 0.2;
  s.addShape(p.ShapeType.ellipse, { x: cxx, y: cyy, w: 0.12, h: 0.12, fill: { color: "94A3B8" } });
}
s.addShape(p.ShapeType.line, { x: dx + 1.7, y: 2.9, w: 0.5, h: 0, line: { color: NAVY, width: 1.5, endArrowType: "triangle" } });
// clusters (3 circles with centroid)
const ccols = [TEAL, "8B5CF6", AMBER];
for (let i = 0; i < 3; i++) {
  const ccx = dx + 2.35, ccy = 2.35 + i * 0.62;
  s.addShape(p.ShapeType.ellipse, { x: ccx, y: ccy, w: 0.5, h: 0.5, fill: { color: ccols[i], transparency: 65 }, line: { color: ccols[i], width: 1 } });
  s.addShape(p.ShapeType.ellipse, { x: ccx + 0.17, y: ccy + 0.17, w: 0.16, h: 0.16, fill: { color: ccols[i] } });
}
s.addText("클러스터 + centroid", { x: dx + 2.05, y: 4.25, w: 1.7, h: 0.5, align: "center", fontFace: F, fontSize: 10, color: MUTED, margin: 0 });
s.addShape(p.ShapeType.line, { x: dx + 2.95, y: 2.9, w: 0.55, h: 0, line: { color: NAVY, width: 1.5, endArrowType: "triangle" } });
// query
s.addShape(p.ShapeType.ellipse, { x: dx + 3.65, y: 2.66, w: 0.4, h: 0.4, fill: { color: INK } });
s.addText("q", { x: dx + 3.65, y: 2.66, w: 0.4, h: 0.4, align: "center", valign: "middle", fontFace: FB, fontSize: 13, bold: true, color: CYAN, margin: 0 });
s.addText("Q·Cᵀ\ntop-k", { x: dx + 3.5, y: 3.15, w: 0.85, h: 0.6, align: "center", fontFace: F, fontSize: 10, color: MUTED, margin: 0 });
s.addShape(p.ShapeType.line, { x: dx + 4.1, y: 2.86, w: 0.45, h: 0, line: { color: TEAL, width: 2, endArrowType: "triangle" } });
s.addShape(p.ShapeType.roundRect, { x: dx + 4.6, y: 2.55, w: 0.75, h: 0.65, rectRadius: 0.06, fill: { color: TEAL } });
s.addText("검색\n결과", { x: dx + 4.6, y: 2.55, w: 0.75, h: 0.65, align: "center", valign: "middle", fontFace: FB, fontSize: 10, bold: true, color: WHITE, margin: 0 });
// budget note
s.addShape(p.ShapeType.roundRect, { x: dx + 0.3, y: 5.15, w: dw - 0.6, h: 1.1, rectRadius: 0.06, fill: { color: "0F172A" } });
s.addText([
  { text: "retrieval_budget", options: { color: TEAL, bold: true } },
  { text: " → 정확 검색할 클러스터 비율\n", options: { color: "E2E8F0" } },
  { text: "estimation_budget", options: { color: AMBER, bold: true } },
  { text: " → 근사 추정할 클러스터 비율", options: { color: "E2E8F0" } },
], { x: dx + 0.5, y: 5.15, w: dw - 1.0, h: 1.1, valign: "middle", fontFace: F, fontSize: 12, lineSpacingMultiple: 1.25, margin: 0 });
footer(s); pageNum(s, 6);

// =========================================================
// Slide 7 — Wave Buffer (light)
// =========================================================
s = p.addSlide();
s.background = { color: WHITE };
title(s, "Wave Buffer: GPU–CPU 협력 실행", "KV cache 배치를 조율하고 연산·데이터 전송을 겹쳐 높은 처리량 유지");
// CPU store
s.addShape(p.ShapeType.roundRect, { x: 0.6, y: 2.1, w: 3.3, h: 3.7, rectRadius: 0.1, fill: { color: CARD }, line: { color: LINE, width: 1 } });
s.addText("CPU 대용량 저장소", { x: 0.6, y: 2.25, w: 3.3, h: 0.4, align: "center", fontFace: FB, fontSize: 14, bold: true, color: NAVY, margin: 0 });
s.addText("WaveBufferCPU (C++/OpenMP)", { x: 0.6, y: 2.62, w: 3.3, h: 0.3, align: "center", fontFace: F, fontSize: 10, color: MUTED, margin: 0 });
for (let i = 0; i < 12; i++) {
  const bx = 0.85 + (i % 4) * 0.72, by = 3.15 + Math.floor(i / 4) * 0.62;
  s.addShape(p.ShapeType.roundRect, { x: bx, y: by, w: 0.6, h: 0.48, rectRadius: 0.04, fill: { color: "CBD5E1" } });
}
s.addText("전체 KV 클러스터 (IVF)", { x: 0.6, y: 5.25, w: 3.3, h: 0.35, align: "center", fontFace: F, fontSize: 11, italic: true, color: "334155", margin: 0 });
// transfer arrows
s.addShape(p.ShapeType.line, { x: 4.0, y: 3.4, w: 1.35, h: 0, line: { color: TEAL, width: 2.5, endArrowType: "triangle" } });
s.addText("gather (miss)", { x: 3.95, y: 3.05, w: 1.5, h: 0.3, align: "center", fontFace: F, fontSize: 10, color: "0F766E", margin: 0 });
s.addShape(p.ShapeType.line, { x: 5.35, y: 4.35, w: 1.35, h: 0, line: { color: NAVY, width: 2.5, endArrowType: "triangle", beginArrowType: "none" } });
s.addText("admit / evict (LRU)", { x: 3.9, y: 4.5, w: 2.9, h: 0.3, align: "center", fontFace: F, fontSize: 10, color: NAVY, margin: 0 });
// GPU buffer
s.addShape(p.ShapeType.roundRect, { x: 5.4, y: 2.1, w: 3.3, h: 3.7, rectRadius: 0.1, fill: { color: "EAF6F4" }, line: { color: TEAL, width: 1.2 } });
s.addText("GPU 실행 버퍼", { x: 5.4, y: 2.25, w: 3.3, h: 0.4, align: "center", fontFace: FB, fontSize: 14, bold: true, color: "0F766E", margin: 0 });
s.addText("BufferManager · LRU 블록 캐시", { x: 5.4, y: 2.62, w: 3.3, h: 0.3, align: "center", fontFace: F, fontSize: 10, color: "0F766E", margin: 0 });
for (let i = 0; i < 6; i++) {
  const bx = 5.75 + (i % 3) * 0.82, by = 3.2 + Math.floor(i / 3) * 0.62;
  const hit = i < 4;
  s.addShape(p.ShapeType.roundRect, { x: bx, y: by, w: 0.68, h: 0.48, rectRadius: 0.04, fill: { color: hit ? TEAL : "A7F3D0" } });
}
s.addText("hit = GPU 캐시 재사용", { x: 5.4, y: 5.25, w: 3.3, h: 0.35, align: "center", fontFace: F, fontSize: 11, italic: true, color: "0F766E", margin: 0 });
// right: benefits
const wb = [
  ["overlap", "연산과 CPU→GPU 전송을 겹쳐 지연 은닉"],
  ["LRU 캐시", "자주 쓰는 클러스터는 GPU에 상주 (hit)"],
  ["스레드풀", "코어 affinity로 CPU 병렬 gather/재조직"],
];
y = 2.15;
wb.forEach((b) => {
  s.addShape(p.ShapeType.roundRect, { x: 9.0, y, w: 3.7, h: 1.15, rectRadius: 0.08, fill: { color: CARD }, line: { color: LINE, width: 1 } });
  s.addText(b[0], { x: 9.25, y: y + 0.15, w: 3.2, h: 0.4, fontFace: FB, fontSize: 14, bold: true, color: NAVY, margin: 0 });
  s.addText(b[1], { x: 9.25, y: y + 0.55, w: 3.25, h: 0.5, fontFace: F, fontSize: 12, color: "334155", lineSpacingMultiple: 1.05, margin: 0 });
  y += 1.28;
});
footer(s); pageNum(s, 7);

// =========================================================
// Slide 8 — Decode flow (light)
// =========================================================
s = p.addSlide();
s.background = { color: WHITE };
title(s, "디코딩 한 스텝의 실행 흐름", "cache_hub/retroinfer_cache.py · sparse_attention()");
const steps = [
  ["batch_gemm_softmax", "Softmax(Q·Cᵀ)로 클러스터 관련도 dist 계산"],
  ["top-k 선택", "빈 클러스터 마스킹 후 nprobe+es 클러스터 선정"],
  ["estimation", "es 클러스터 centroid로 근사 (es_out, es_lse)"],
  ["wave_buffer.access", "batch_access로 CPU→GPU 이동을 스레드풀에 제출"],
  ["gather + concat", "steady+retrieval KV를 execution buffer에 조립"],
  ["weighted_flash_decoding", "정확 어텐션 + estimation online-softmax 병합"],
  ["scatter admit", "사용 페이지를 GPU 블록 캐시에 admit (LRU)"],
];
// 7 steps as connected cards, 4 top row + 3 bottom row
const bw = 2.85, bh = 1.35, gapx = 0.22;
function stepCard(x, yy, n, t, d, hl) {
  s.addShape(p.ShapeType.roundRect, { x, y: yy, w: bw, h: bh, rectRadius: 0.08, fill: { color: hl ? "EAF6F4" : CARD }, line: { color: hl ? TEAL : LINE, width: hl ? 1.3 : 1 } });
  s.addShape(p.ShapeType.ellipse, { x: x + 0.18, y: yy + 0.18, w: 0.42, h: 0.42, fill: { color: hl ? TEAL : NAVY } });
  s.addText(String(n), { x: x + 0.18, y: yy + 0.18, w: 0.42, h: 0.42, align: "center", valign: "middle", fontFace: FB, fontSize: 14, bold: true, color: WHITE, margin: 0 });
  s.addText(t, { x: x + 0.7, y: yy + 0.16, w: bw - 0.85, h: 0.46, fontFace: "Consolas", fontSize: 11, bold: true, color: hl ? "0F766E" : TEXT, margin: 0 });
  s.addText(d, { x: x + 0.22, y: yy + 0.68, w: bw - 0.4, h: 0.6, fontFace: F, fontSize: 10.5, color: "334155", lineSpacingMultiple: 1.02, margin: 0 });
}
const topY = 2.05, botY = 4.35;
for (let i = 0; i < 4; i++) {
  const x = 0.6 + i * (bw + gapx);
  stepCard(x, topY, i + 1, steps[i][0], steps[i][1], i === 5);
  if (i < 3) s.addShape(p.ShapeType.line, { x: x + bw, y: topY + bh / 2, w: gapx, h: 0, line: { color: NAVY, width: 2, endArrowType: "triangle" } });
}
// down arrow from step4 to step5 area
s.addShape(p.ShapeType.line, { x: 0.6 + 3 * (bw + gapx) + bw / 2, y: topY + bh, w: 0, h: (botY - (topY + bh)) / 1, line: { color: NAVY, width: 2, endArrowType: "triangle" } });
// bottom row 5,6,7 right-to-left visually; place left to right as 5,6,7 under
const botOrder = [4, 5, 6];
for (let k = 0; k < 3; k++) {
  const i = botOrder[k];
  const x = 0.6 + (3 - k) * (bw + gapx); // right to left
  stepCard(x, botY, i + 1, steps[i][0], steps[i][1], i === 5);
  if (k < 2) s.addShape(p.ShapeType.line, { x: x, y: botY + bh / 2, w: -gapx, h: 0, line: { color: NAVY, width: 2, endArrowType: "triangle" } });
}
s.addText("→  결과를 다음 레이어로 전달, 매 디코딩 스텝 반복", { x: 0.6, y: 6.35, w: 8, h: 0.35, fontFace: F, fontSize: 12, italic: true, color: MUTED, margin: 0 });
footer(s); pageNum(s, 8);

// =========================================================
// Slide 9 — CUDA kernels (light)
// =========================================================
s = p.addSlide();
s.background = { color: WHITE };
title(s, "고성능 CUDA / C++ 커널", "library/retroinfer/retroinfer_kernels — GPU–CPU 데이터 이동과 융합 연산");
const ks = [
  ["batch_gemm_softmax.cu", NAVY, "CUTLASS 융합 GEMM + Softmax", [
    "Q·Cᵀ와 2단계 online-softmax를 융합",
    "InstructionShape ⟨16,8,16⟩ · Sm80 Tensor Core",
    "bf16 / fp16 지원",
  ]],
  ["gather_copy.cu / .cuh", TEAL, "wave buffer 데이터 이동", [
    "gather / scatter / concat / reorganize 5종",
    "int2 벡터화 접근 · 동적 블록 분배",
    "steady+retrieval KV 조립, LRU admit",
  ]],
  ["wave_buffer_cpu.cpp", "8B5CF6", "CPU 측 협력 로직", [
    "LRU BufferManager (hit/miss/admit/evict)",
    "클러스터를 IVF로 재조직",
    "코어 affinity 스레드풀 병렬화",
  ]],
];
ks.forEach((k, i) => {
  const cx = 0.6 + i * 4.15;
  s.addShape(p.ShapeType.roundRect, { x: cx, y: 1.95, w: 3.85, h: 4.2, rectRadius: 0.1, fill: { color: CARD }, line: { color: LINE, width: 1 } });
  s.addShape(p.ShapeType.roundRect, { x: cx, y: 1.95, w: 3.85, h: 0.95, rectRadius: 0.1, fill: { color: k[1] } });
  s.addShape(p.ShapeType.rect, { x: cx, y: 2.55, w: 3.85, h: 0.35, fill: { color: k[1] } });
  s.addText(k[0], { x: cx + 0.25, y: 2.06, w: 3.4, h: 0.4, fontFace: "Consolas", fontSize: 12.5, bold: true, color: WHITE, margin: 0 });
  s.addText(k[2], { x: cx + 0.25, y: 2.46, w: 3.4, h: 0.4, fontFace: F, fontSize: 11.5, color: "EAF2FF", margin: 0 });
  const items = k[3].map((t, idx) => ({ text: t, options: { bullet: { code: "2022", indent: 12 }, breakLine: idx < k[3].length - 1, paraSpaceAfter: 8 } }));
  s.addText(items, { x: cx + 0.3, y: 3.1, w: 3.3, h: 2.9, fontFace: F, fontSize: 12, color: "1E293B", lineSpacingMultiple: 1.05, margin: 0, valign: "top" });
});
footer(s); pageNum(s, 9);

// =========================================================
// Slide 10 — Results (light + chart)
// =========================================================
s = p.addSlide();
s.background = { color: WHITE };
title(s, "성능: 정확도 유지, 처리량 4.5–10.5×", "FlashAttention 대비 디코딩 처리량 향상 (논문 보고치)");
// chart
const chartData = [{
  name: "상대 처리량",
  labels: ["FlashAttention", "RetroInfer (하한)", "RetroInfer (상한)"],
  values: [1.0, 4.5, 10.5],
}];
s.addChart(p.ChartType.bar, chartData, {
  x: 0.6, y: 2.0, w: 7.0, h: 4.2,
  barDir: "bar",
  chartColors: [TEAL],
  showTitle: false, showLegend: false,
  showValue: true, dataLabelPosition: "outEnd",
  dataLabelColor: TEXT, dataLabelFontFace: F, dataLabelFontSize: 12, dataLabelFontBold: true,
  dataLabelFormatCode: '0.0"×"',
  catAxisLabelColor: TEXT, catAxisLabelFontFace: F, catAxisLabelFontSize: 12,
  valAxisHidden: true, valGridLine: { style: "none" },
  catGridLine: { style: "none" },
  valAxisMaxVal: 12, valAxisMinVal: 0,
  barGapWidthPct: 60,
});
s.addText("* 값은 논문 보고 범위(4.5–10.5×)를 도식화한 것", { x: 0.6, y: 6.25, w: 7, h: 0.3, fontFace: F, fontSize: 10, italic: true, color: MUTED, margin: 0 });
// right stat callouts
s.addShape(p.ShapeType.roundRect, { x: 8.0, y: 2.0, w: 4.7, h: 1.95, rectRadius: 0.1, fill: { color: INK } });
s.addText("4.5–10.5×", { x: 8.2, y: 2.2, w: 4.3, h: 0.95, fontFace: FB, fontSize: 48, bold: true, color: TEAL, margin: 0 });
s.addText("디코딩 처리량 향상 (정확도 손실 없이)", { x: 8.2, y: 3.2, w: 4.3, h: 0.5, fontFace: F, fontSize: 13, color: "CADCFC", margin: 0 });
s.addShape(p.ShapeType.roundRect, { x: 8.0, y: 4.1, w: 4.7, h: 2.1, rectRadius: 0.1, fill: { color: CARD }, line: { color: LINE, width: 1 } });
s.addText("정확도 검증", { x: 8.25, y: 4.25, w: 4.2, h: 0.4, fontFace: FB, fontSize: 14, bold: true, color: NAVY, margin: 0 });
const acc = ["RULER (128K) — NIAH·VT·CWE·FWE·QA", "LongBench — SQA·MQA·요약·코드", "AIME / GPQA 장문 추론"];
s.addText(acc.map((t, i) => ({ text: t, options: { bullet: { code: "2022", indent: 12 }, breakLine: i < acc.length - 1, paraSpaceAfter: 8 } })),
  { x: 8.35, y: 4.7, w: 4.1, h: 1.35, fontFace: F, fontSize: 12, color: "1E293B", margin: 0, valign: "top" });
footer(s); pageNum(s, 10);

// =========================================================
// Slide 11 — Requirements & limits (light)
// =========================================================
s = p.addSlide();
s.background = { color: WHITE };
title(s, "하드웨어 요구사항과 한계", "핵심 커널이 Ampere Tensor Core를 하드코딩 → 구형 GPU 불가");
// requirement table (left)
const reqs = [
  ["GPU 아키텍처", "Ampere(sm_80)+ Tensor Core", true],
  ["데이터 타입", "bfloat16 / float16", true],
  ["CUDA", "12.4 (cu124)", true],
  ["핵심 의존성", "flash-attn 2.7.3 · flashinfer · CUTLASS", true],
];
s.addText("요구사항", { x: 0.6, y: 1.95, w: 6, h: 0.4, fontFace: FB, fontSize: 16, bold: true, color: "0F766E", margin: 0 });
y = 2.5;
reqs.forEach((r) => {
  s.addShape(p.ShapeType.roundRect, { x: 0.6, y, w: 5.9, h: 0.82, rectRadius: 0.06, fill: { color: "EAF6F4" }, line: { color: TEAL, width: 1 } });
  s.addText(r[0], { x: 0.85, y, w: 2.1, h: 0.82, valign: "middle", fontFace: FB, fontSize: 12.5, bold: true, color: "134E4A", margin: 0 });
  s.addText(r[1], { x: 2.95, y, w: 3.4, h: 0.82, valign: "middle", fontFace: F, fontSize: 12, color: "1E293B", margin: 0 });
  y += 0.95;
});
// limits (right)
s.addText("한계 / 미지원", { x: 6.9, y: 1.95, w: 6, h: 0.4, fontFace: FB, fontSize: 16, bold: true, color: "B45309", margin: 0 });
const lims = [
  ["Pascal 등 구형 GPU 불가", "Sm80 MMA·Tensor Core 요구. Quadro P6000(sm_61) 등에서 컴파일/실행 불가"],
  ["bf16 하드웨어 의존", "Ampere 이전은 bf16 미지원 (fp16 전환해도 커널 아키텍처 벽이 남음)"],
  ["GPU 없이 실행 불가", "커스텀 커널·flash-attn이 CUDA 컴파일 코드 — CPU 시뮬레이션 경로 없음"],
];
y = 2.5;
lims.forEach((r) => {
  s.addShape(p.ShapeType.roundRect, { x: 6.9, y, w: 5.8, h: 1.15, rectRadius: 0.06, fill: { color: "FEF3E2" }, line: { color: AMBER, width: 1 } });
  s.addText(r[0], { x: 7.15, y: y + 0.13, w: 5.3, h: 0.4, fontFace: FB, fontSize: 13, bold: true, color: "92400E", margin: 0 });
  s.addText(r[1], { x: 7.15, y: y + 0.5, w: 5.35, h: 0.6, fontFace: F, fontSize: 11.5, color: "5B4322", lineSpacingMultiple: 1.05, margin: 0 });
  y += 1.28;
});
footer(s); pageNum(s, 11);

// ---- code box helper ----
function codeBox(s, x, y, w, h, text, fs) {
  s.addShape(p.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.06, fill: { color: "0F172A" } });
  s.addText(text, { x: x + 0.18, y: y + 0.12, w: w - 0.36, h: h - 0.24, fontFace: "Consolas", fontSize: fs || 10.5, color: "E2E8F0", lineSpacingMultiple: 1.14, margin: 0, valign: "top" });
}
function colHead(s, x, y, w, t, col) {
  s.addText(t, { x, y, w, h: 0.36, fontFace: FB, fontSize: 15, bold: true, color: col || NAVY, margin: 0 });
}

// =========================================================
// Slide 12 — 실전 ① 환경 설정 & 커널 설치 (light)
// =========================================================
s = p.addSlide();
s.background = { color: WHITE };
title(s, "실전 ① 환경 설정 & 커널 설치", "CUDA 12.4 · Python 3.10.16 · conda 권장 (README.kr.md)");
colHead(s, 0.6, 1.95, 6.0, "1. 패키지 설치", "0F766E");
codeBox(s, 0.6, 2.35, 6.0, 3.15,
  "conda create -n retroinfer python=3.10 -y\n" +
  "conda activate retroinfer\n" +
  "conda install -y mkl\n" +
  "conda install -c conda-forge libstdcxx-ng -y\n\n" +
  "pip install -r requirements.txt\n" +
  "pip install flash-attn==2.7.3 \\\n" +
  "    --no-build-isolation\n" +
  "pip install flashinfer-python==0.2.4 \\\n" +
  "    -i https://flashinfer.ai/whl/cu124/torch2.5/\n" +
  "pip install git+.../Starmys/\\\n" +
  "    flash-attention.git@weighted", 10.5);
colHead(s, 6.9, 1.95, 5.8, "2. 커널 빌드", NAVY);
codeBox(s, 6.9, 2.35, 5.8, 3.15,
  "cd library/\n" +
  "git clone \\\n" +
  "  https://github.com/NVIDIA/cutlass.git\n" +
  "cd retroinfer && pip install . && cd ..\n\n" +
  "# 선택: MInference prefill\n" +
  "pip install minference==0.1.6.0\n\n" +
  "# 선택: XAttention prefill\n" +
  "git clone .../Block-Sparse-Attention.git\n" +
  "cd Block-Sparse-Attention && \\\n" +
  "  python setup.py install", 10.5);
s.addShape(p.ShapeType.roundRect, { x: 0.6, y: 5.7, w: 12.1, h: 0.62, rectRadius: 0.06, fill: { color: "FEF3E2" }, line: { color: AMBER, width: 1 } });
s.addText([
  { text: "TIP  ", options: { bold: true, color: "92400E" } },
  { text: "CUDA 12.4 미설치 시 도커 이미지 사용: nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04", options: { color: "5B4322" } },
], { x: 0.85, y: 5.7, w: 11.6, h: 0.62, valign: "middle", fontFace: F, fontSize: 12, margin: 0 });
footer(s); pageNum(s, 12);

// =========================================================
// Slide 13 — 실전 ② 데모 실행 & API (light)
// =========================================================
s = p.addSlide();
s.background = { color: WHITE };
title(s, "실전 ② 데모 실행 & API", "환경 검증용 데모와 Python API 진입점");
colHead(s, 0.6, 1.95, 6.0, "빠른 데모 — simple_test.py", "0F766E");
codeBox(s, 0.6, 2.35, 6.0, 1.15,
  "python -u simple_test.py \\\n" +
  "       --batch_size 4", 12);
s.addText([
  { text: "메모리: ", options: { bold: true, color: TEXT } },
  { text: "약 35GB GPU + 70GB CPU (batch 4)", options: { color: "334155" } },
], { x: 0.62, y: 3.62, w: 6.0, h: 0.35, fontFace: F, fontSize: 12, margin: 0 });
const opts = [
  ["--gpu_only", "전체 KV를 GPU에 상주"],
  ["--use_cuda_graph", "CUDA Graph로 런치 오버헤드 감소"],
  ["--do_sample", "샘플링 디코딩"],
  ["--prefill_method", "full / xattn / minfer"],
];
y = 4.05;
opts.forEach((o) => {
  s.addText([
    { text: o[0] + "  ", options: { fontFace: "Consolas", bold: true, color: "0F766E" } },
    { text: o[1], options: { fontFace: F, color: "334155" } },
  ], { x: 0.7, y, w: 5.9, h: 0.36, fontSize: 12, margin: 0 });
  y += 0.5;
});
colHead(s, 6.9, 1.95, 5.8, "Python API", NAVY);
codeBox(s, 6.9, 2.35, 5.8, 3.95,
  "from model_hub import load_model, \\\n" +
  "     load_tokenizer\n" +
  "from config import generate_config\n\n" +
  "tokenizer = load_tokenizer(model_name)\n" +
  "llm = load_model(model_name, max_seq_len,\n" +
  "                 dtype, device, tokenizer)\n\n" +
  "attn_config = generate_config(\n" +
  "    model_name, input_seq_len, \"RetroInfer\",\n" +
  "    retrieval_budget, estimation_budget,\n" +
  "    cache_ratio)\n\n" +
  "out = llm.generate(\n" +
  "    attention_type=attn_type,\n" +
  "    inputs_ids=input_ids, ...,\n" +
  "    attn_config=attn_config)", 10.5);
footer(s); pageNum(s, 13);

// =========================================================
// Slide 14 — 실전 ③ 정확도 벤치마크 (light)
// =========================================================
s = p.addSlide();
s.background = { color: WHITE };
title(s, "실전 ③ 정확도 벤치마크", "긴 컨텍스트 표준 벤치마크로 정확도 검증");
const bms = [
  ["RULER", "8B5CF6", "bash ruler_run.sh llama-3-8b-1048k \\\n  full RetroInfer 131072 vt bf16 \\\n  0.018 0.232",
    "128K 컨텍스트 · NIAH/VT/CWE/FWE/QA\n인자: 모델·prefill·attn·길이·태스크·dtype·budget"],
  ["LongBench", TEAL, "bash longbench_run.sh \\\n  llama-3-8b-1048k RetroInfer \\\n  0.018 0.232 bf16 SQA",
    "SQA·MQA·SUM·FSL·ST·CC 범주\n인자: 모델·attn·budget·dtype·범주"],
  ["Reasoning", "F59E0B", "bash eval.sh \\\n  deepseek-ai/DeepSeek-R1-\\\n  Distill-Llama-8B RetroInfer \\\n  aime24 0 -1",
    "AIME / GPQA 장문 추론 (pass@k)\n인자: 모델·attn·데이터·시작·개수"],
];
bms.forEach((b, i) => {
  const cx = 0.6 + i * 4.15;
  s.addShape(p.ShapeType.roundRect, { x: cx, y: 1.95, w: 3.85, h: 4.35, rectRadius: 0.1, fill: { color: CARD }, line: { color: LINE, width: 1 } });
  s.addShape(p.ShapeType.roundRect, { x: cx, y: 1.95, w: 3.85, h: 0.62, rectRadius: 0.1, fill: { color: b[1] } });
  s.addShape(p.ShapeType.rect, { x: cx, y: 2.35, w: 3.85, h: 0.22, fill: { color: b[1] } });
  s.addText(b[0], { x: cx + 0.25, y: 1.95, w: 3.35, h: 0.62, valign: "middle", fontFace: FB, fontSize: 17, bold: true, color: WHITE, margin: 0 });
  codeBox(s, cx + 0.22, y = 2.75, 3.4, 1.95, b[2], 9);
  s.addText(b[3], { x: cx + 0.25, y: 4.85, w: 3.4, h: 1.3, fontFace: F, fontSize: 11, color: "334155", lineSpacingMultiple: 1.1, margin: 0 });
});
footer(s); pageNum(s, 14);

// =========================================================
// Slide 15 — 실전 ④ 처리량 재현 & 확장 (light)
// =========================================================
s = p.addSlide();
s.background = { color: WHITE };
title(s, "실전 ④ 처리량 재현 & 새 방법 추가", "논문 처리량 재현과 프레임워크 확장");
colHead(s, 0.6, 1.95, 6.0, "처리량 재현", "0F766E");
codeBox(s, 0.6, 2.35, 6.0, 1.7,
  "sudo apt install numactl -y\n\n" +
  "cd throughput_eval\n" +
  "bash run.sh", 12);
s.addShape(p.ShapeType.roundRect, { x: 0.6, y: 4.2, w: 6.0, h: 1.4, rectRadius: 0.08, fill: { color: "EAF6F4" }, line: { color: TEAL, width: 1 } });
s.addText([
  { text: "실험 환경 (논문)\n", options: { bold: true, color: "0F766E" } },
  { text: "Azure 4-NUMA A100 머신 · NUMA 노드당 24코어 · 475GB CPU · 80GB A100 ×2", options: { color: "134E4A" } },
], { x: 0.85, y: 4.35, w: 5.5, h: 1.1, fontFace: F, fontSize: 12, lineSpacingMultiple: 1.15, margin: 0, valign: "top" });
colHead(s, 6.9, 1.95, 5.8, "새 희소성 방법 추가 — 5단계", NAVY);
const ext = [
  ["cache_hub/", "KV 캐시 관리 로직 추가"],
  ["attn_hub/", "어텐션 연산 로직 추가"],
  ["config/config.py", "설정 옵션 등록"],
  ["model_hub/llama.py · qwen.py", "init_kv_cache · decode_attention · parameter_move 갱신"],
  ["--attn_type", "지정해 실행"],
];
y = 2.4;
ext.forEach((e, i) => {
  chip(s, 6.9, y, String(i + 1), NAVY);
  s.addText([
    { text: e[0] + "  ", options: { fontFace: "Consolas", bold: true, color: "0F766E" } },
    { text: e[1], options: { fontFace: F, color: "334155" } },
  ], { x: 7.5, y: y + 0.02, w: 5.2, h: 0.5, fontSize: 11.5, lineSpacingMultiple: 1.0, margin: 0, valign: "middle" });
  y += 0.78;
});
footer(s); pageNum(s, 15);

// =========================================================
// Slide 16 — Summary (dark)
// =========================================================
s = p.addSlide();
s.background = { color: INK };
s.addShape(p.ShapeType.rect, { x: 0, y: H - 0.14, w: W, h: 0.14, fill: { color: TEAL } });
s.addText("요약", { x: 0.6, y: 0.55, w: 6, h: 0.6, fontFace: FB, fontSize: 30, bold: true, color: WHITE, margin: 0 });
const takeaways = [
  ["벡터 저장 엔진으로의 재해석", "KV 캐시를 ANN 인덱스로 다뤄 관련 토큰만 검색 → 긴 컨텍스트에서 계산량 폭증 회피"],
  ["정확도 보장 3-zone 어텐션", "steady·retrieval·estimation을 online-softmax로 병합 → 정확도 손실 없이 희소화"],
  ["GPU–CPU 협력 + 융합 커널", "wave buffer로 전송·연산 overlap, CUTLASS/Tensor Core로 4.5–10.5× 처리량"],
];
y = 1.6;
takeaways.forEach((t, i) => {
  s.addShape(p.ShapeType.roundRect, { x: 0.6, y, w: 12.1, h: 1.25, rectRadius: 0.09, fill: { color: INK2 }, line: { color: "27406E", width: 1 } });
  s.addShape(p.ShapeType.ellipse, { x: 0.85, y: y + 0.32, w: 0.6, h: 0.6, fill: { color: TEAL } });
  s.addText(String(i + 1), { x: 0.85, y: y + 0.32, w: 0.6, h: 0.6, align: "center", valign: "middle", fontFace: FB, fontSize: 22, bold: true, color: INK, margin: 0 });
  s.addText(t[0], { x: 1.7, y: y + 0.2, w: 10.7, h: 0.45, fontFace: FB, fontSize: 17, bold: true, color: WHITE, margin: 0 });
  s.addText(t[1], { x: 1.7, y: y + 0.65, w: 10.8, h: 0.5, fontFace: F, fontSize: 13, color: "AEBEDD", margin: 0 });
  y += 1.4;
});
s.addText([
  { text: "참고문헌   ", options: { bold: true, color: TEAL } },
  { text: "RetroInfer (VLDB 2026, arXiv:2505.02922)  ·  RetrievalAttention (arXiv:2409.10516)", options: { color: "8AA0C8" } },
], { x: 0.6, y: 6.35, w: 12.1, h: 0.4, fontFace: F, fontSize: 12, margin: 0 });
pageNum(s, 16, true);

p.writeFile({ fileName: "RetroInfer_세미나.pptx" })
  .then(f => console.log("WROTE", f))
  .catch(e => { console.error(e); process.exit(1); });
