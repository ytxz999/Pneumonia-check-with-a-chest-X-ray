/*!
 * CircularCarousel —— React Bits「CircularCarousel」组件的原生 JS (vanilla) 移植版
 * 来源: https://reactbits.dev  (CircularCarousel, JavaScript + CSS variant)
 * 移植说明: 去除 React 依赖；3D 环形布局 / 弹簧物理 / 拖拽惯性 / 深度淡出 /
 *   入场动画 / 数字滚动计数等逻辑与原版对应。省略了 items 动态更换的 crossfade。
 *
 * 用法:
 *   TechCarousel.create(el, {
 *     items: [{ src, title, subtitle }, ...],
 *     preset: 'cylinder', intro: 'rise', cardWidth: 220,
 *     aspectRatio: 1, speed: 14, captions: true, fadeColor: '#070b16'
 *   });
 */
(() => {
  'use strict';

  const TILES = 8;
  const OVERLAP = 2.5;
  const DRAG_THRESHOLD = 5;
  const SPRING = 118;
  const SETTLE_SPEED = 9;
  const CAPTION_SPACE = 76;
  const TO_RAD = Math.PI / 180;
  const INTRO_LENGTH = { assemble: 1500, rise: 1400, spin: 1800, none: 0 };

  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const wrap = d => ((((d + 180) % 360) + 360) % 360) - 180;
  const easeOut = t => 1 - Math.pow(1 - t, 4);
  const easeOutQuint = t => 1 - Math.pow(1 - t, 5);

  const PRESETS = {
    cylinder:  { axis: 'y', tilt: -5,  perspective: 2500, curve: 1, spread: 1,    inward: false, billboard: false, backfaces: true,  window: 0 },
    orbit:     { axis: 'y', tilt: -16, perspective: 1500, curve: 0, spread: 1.45, inward: false, billboard: true,  backfaces: false, window: 0 },
    wheel:     { axis: 'x', tilt: 0,   perspective: 1800, curve: 0, spread: 1,    inward: false, billboard: false, backfaces: true,  window: 1.7 },
    panorama:  { axis: 'y', tilt: 0,   perspective: 0,    curve: 1, spread: 1,    inward: true,  billboard: false, backfaces: false, window: 0 }
  };

  const CSS = `
.circular-carousel{--cc-fade:#000000;--cc-radius:12px;--cc-inner:.4;position:relative;width:100%;height:100%;overflow:hidden;outline:none;touch-action:pan-y;user-select:none;-webkit-user-select:none;-webkit-tap-highlight-color:transparent}
.circular-carousel[data-draggable]{cursor:grab}
.circular-carousel[data-dragging]{cursor:grabbing}
.circular-carousel:focus-visible{box-shadow:inset 0 0 0 2px rgba(128,140,180,.55)}
.circular-carousel__view{position:absolute;inset:0}
.circular-carousel__stage{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;opacity:0;transition:opacity .45s ease}
.circular-carousel[data-ready] .circular-carousel__stage{opacity:1}
.circular-carousel__camera,.circular-carousel__ring,.circular-carousel__card{width:0;height:0;transform-style:preserve-3d}
.circular-carousel__camera{position:relative}
.circular-carousel__ring,.circular-carousel__card{position:absolute;left:0;top:0}
.circular-carousel__tile{position:absolute;outline:1px solid transparent;backface-visibility:hidden;-webkit-backface-visibility:hidden}
.circular-carousel__frame{position:absolute;left:0;top:0;width:100%;overflow:hidden}
.circular-carousel__photo{position:absolute;display:block;max-width:none;object-fit:cover;pointer-events:none;user-select:none;-webkit-user-drag:none}
.circular-carousel__shade,.circular-carousel__inner{position:absolute;inset:0;pointer-events:none}
.circular-carousel__shade{background:var(--cc-fade);opacity:var(--cc-depth,0)}
.circular-carousel__inner{background:#000;opacity:var(--cc-inner)}
.circular-carousel__caption{position:absolute;left:0;right:0;bottom:22px;display:flex;flex-direction:column;align-items:center;gap:6px;padding:0 16px;text-align:center;pointer-events:none;animation:cc-reveal .9s ease both}
.circular-carousel__title{display:flex;max-width:100%;flex-direction:column;align-items:center;font-size:15px;font-weight:500;line-height:1.35;animation:cc-title .52s cubic-bezier(.22,1,.36,1)}
.circular-carousel__subtitle{font-weight:400;opacity:.6}
.circular-carousel__count{display:inline-flex;align-items:center;gap:3px;font-size:12px;font-variant-numeric:tabular-nums;line-height:1;opacity:.5}
.circular-carousel__slash{opacity:.6}
.circular-carousel__digits{display:inline-flex}
.circular-carousel__digit{display:inline-block;height:1em;overflow:hidden}
.circular-carousel__reel{display:flex;flex-direction:column;transition:transform .6s cubic-bezier(.22,1,.36,1)}
.circular-carousel__reel>span{height:1em}
.circular-carousel__live{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
@keyframes cc-title{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
@keyframes cc-reveal{from{opacity:0}to{opacity:1}}
@media (prefers-reduced-motion: reduce){
  .circular-carousel__stage,.circular-carousel__reel{transition:none}
  .circular-carousel__caption,.circular-carousel__title{animation:none}
}`;

  const DEFAULTS = {
    preset: 'cylinder', intro: 'rise', cardWidth: 220, aspectRatio: 1, gap: 25,
    autoplay: 'drift', speed: 14, interval: 3, direction: 'left',
    draggable: true, momentum: 0.6, snap: true, pauseOnHover: true,
    focusOnClick: true, parallax: 0.3, stretch: 0.5, depthFade: 0.55,
    fadeColor: '#000000', innerShade: 0.6, cornerRadius: 12, captions: false,
    curve: undefined, tilt: undefined, perspective: undefined
  };

  function injectCss() {
    if (document.getElementById('circular-carousel-css')) return;
    const style = document.createElement('style');
    style.id = 'circular-carousel-css';
    style.textContent = CSS;
    document.head.appendChild(style);
  }

  function create(root, userOpts) {
    const o = Object.assign({}, DEFAULTS, userOpts || {});
    const items = (o.items && o.items.length ? o.items : []);
    if (!items.length) return null;
    injectCss();

    const reduced =
      window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches || false;

    /* ---------- DOM ---------- */
    root.classList.add('circular-carousel');
    root.setAttribute('role', 'region');
    root.setAttribute('aria-roledescription', 'carousel');
    root.tabIndex = 0;
    const view = document.createElement('div');
    view.className = 'circular-carousel__view';
    const stage = document.createElement('div');
    stage.className = 'circular-carousel__stage';
    const camera = document.createElement('div');
    camera.className = 'circular-carousel__camera';
    const ring = document.createElement('div');
    ring.className = 'circular-carousel__ring';
    camera.appendChild(ring);
    stage.appendChild(camera);
    view.appendChild(stage);
    root.appendChild(view);

    root.style.setProperty('--cc-fade', o.fadeColor);
    root.style.setProperty('--cc-radius', `${Math.max(0, o.cornerRadius)}px`);
    root.style.setProperty('--cc-inner', (1 - clamp(o.innerShade, 0, 1)).toFixed(3));
    if (o.draggable) root.setAttribute('data-draggable', '');

    /* ---------- 布局参数 ---------- */
    const layout = PRESETS[o.preset] ? PRESETS[o.preset] : PRESETS.cylinder;
    const axis = layout.axis;
    const tiltValue = o.tilt ?? layout.tilt;
    const curveValue = layout.billboard ? 0 : clamp(o.curve ?? layout.curve, 0, 1);
    const count = items.length;
    const cardW = Math.max(40, o.cardWidth);
    const cardH = cardW / clamp(o.aspectRatio, 0.2, 5);
    const along = axis === 'x' ? cardH : cardW;
    const step = 360 / count;
    const n = Math.max(count, 3);
    const pitch = (along + o.gap) * layout.spread;
    const chord = pitch / (2 * Math.sin(Math.PI / n));
    const arc = (n * pitch) / (2 * Math.PI);
    const radius = Math.max(chord + (arc - chord) * curveValue, along * 0.6);
    const perspective = layout.inward ? radius : (o.perspective ?? layout.perspective);
    const captions = o.captions;
    const dragSign = layout.inward ? -1 : 1;
    const dir0 = (o.direction === 'right' ? 1 : -1) * dragSign;

    /* ---------- 卡片 DOM（分片弯曲） ---------- */
    const total = curveValue > 0.001 ? TILES : 1;
    const length = along / total;
    const bend = curveValue > 0.001 ? radius / curveValue : 0;
    const tiles = Array.from({ length: total }, (_, index) => {
      const start = index * length - (index > 0 ? OVERLAP / 2 : 0);
      const end = (index + 1) * length + (index < total - 1 ? OVERLAP / 2 : 0);
      const center = (start + end) / 2 - along / 2;
      const alpha = bend ? center / bend : 0;
      const shift = bend ? bend * Math.sin(alpha) : center;
      const sink = bend ? bend * (1 - Math.cos(alpha)) : 0;
      const depth = layout.inward ? sink : -sink;
      const turn = ((layout.inward ? -alpha : alpha) * 180) / Math.PI;
      const move = axis === 'x'
        ? `translate3d(0px, ${shift}px, ${depth}px) rotateX(${-turn}deg)`
        : `translate3d(${shift}px, 0px, ${depth}px) rotateY(${turn}deg)`;
      return { index, total, start, end, size: end - start, move };
    });

    const cardRefs = [];
    items.forEach((item, index) => {
      const card = document.createElement('div');
      card.className = 'circular-carousel__card';
      card.setAttribute('data-cc-index', index);
      card.setAttribute('role', 'group');
      card.setAttribute('aria-roledescription', 'slide');
      card.setAttribute('aria-label',
        `${item.title || item.alt || `Image ${index + 1}`}, ${index + 1} / ${count}`);
      for (const back of (layout.backfaces ? [false, true] : [false])) {
        for (const tile of tiles) {
          const strip = back ? tile.total - 1 - tile.index : tile.index;
          const first = strip === 0;
          const last = strip === tile.total - 1;
          const r = 'var(--cc-radius)';
          const frameRadius = axis === 'x'
            ? `${first ? r : 0} ${first ? r : 0} ${last ? r : 0} ${last ? r : 0}`
            : `${first ? r : 0} ${last ? r : 0} ${last ? r : 0} ${first ? r : 0}`;
          const offset = back ? along - tile.end : tile.start;
          const box = axis === 'x'
            ? { left: -cardW / 2, top: -tile.size / 2, width: cardW, height: tile.size }
            : { left: -tile.size / 2, top: -cardH / 2, width: tile.size, height: cardH };
          const photoStyle = axis === 'x'
            ? { left: 0, top: -offset, width: cardW, height: cardH }
            : { left: -offset, top: 0, width: cardW, height: cardH };
          const el = document.createElement('div');
          el.className = 'circular-carousel__tile';
          el.setAttribute('aria-hidden', 'true');
          el.style.cssText =
            `left:${box.left}px;top:${box.top}px;width:${box.width}px;height:${box.height}px;` +
            `transform:${tile.move}${back ? (axis === 'x' ? ' rotateX(180deg)' : ' rotateY(180deg)') : ''}`;
          const frame = document.createElement('div');
          frame.className = 'circular-carousel__frame';
          frame.style.height = (axis === 'x' ? tile.size : cardH) + 'px';
          frame.style.borderRadius = frameRadius;
          const img = document.createElement('img');
          img.className = 'circular-carousel__photo';
          img.src = item.src;
          img.alt = '';
          img.draggable = false;
          img.decoding = 'async';
          img.style.cssText =
            `left:${photoStyle.left}px;top:${photoStyle.top}px;` +
            `width:${photoStyle.width}px;height:${photoStyle.height}px`;
          frame.appendChild(img);
          if (back) {
            const inner = document.createElement('div');
            inner.className = 'circular-carousel__inner';
            frame.appendChild(inner);
          }
          const shade = document.createElement('div');
          shade.className = 'circular-carousel__shade';
          frame.appendChild(shade);
          el.appendChild(frame);
          card.appendChild(el);
        }
      }
      ring.appendChild(card);
      cardRefs.push(card);
    });

    /* ---------- captions ---------- */
    let capTitle = null, capDigits = null, capCount = null;
    if (captions) {
      const cap = document.createElement('div');
      cap.className = 'circular-carousel__caption';
      cap.setAttribute('aria-hidden', 'true');
      capTitle = document.createElement('span');
      capTitle.className = 'circular-carousel__title';
      const cnt = document.createElement('span');
      cnt.className = 'circular-carousel__count';
      capDigits = document.createElement('span');
      capDigits.className = 'circular-carousel__digits';
      const pad2 = v => String(v).padStart(2, '0');
      for (const d of pad2(1)) {
        const digit = document.createElement('span');
        digit.className = 'circular-carousel__digit';
        const reel = document.createElement('span');
        reel.className = 'circular-carousel__reel';
        reel.style.transform = `translateY(${-Number(d) * 10}%)`;
        for (const c of '0123456789') {
          const s = document.createElement('span');
          s.textContent = c;
          reel.appendChild(s);
        }
        digit.appendChild(reel);
        capDigits.appendChild(digit);
      }
      const slash = document.createElement('span');
      slash.className = 'circular-carousel__slash';
      slash.textContent = '/';
      capCount = document.createElement('span');
      capCount.textContent = pad2(count);
      cnt.append(capDigits, slash, capCount);
      cap.append(capTitle, cnt);
      root.appendChild(cap);
    }
    const live = document.createElement('div');
    live.className = 'circular-carousel__live';
    live.setAttribute('aria-live', 'polite');
    live.setAttribute('aria-atomic', 'true');
    root.appendChild(live);

    /* ---------- 状态 ---------- */
    const state = {
      angle: 0, velocity: 0, target: null, dir: dir0,
      press: null, drag: false, hover: false,
      pointer: { inside: false, x: 0, y: 0 },
      yaw: 0, pitch: 0, intro: null, introDone: false,
      holdUntil: 0, wheelTimer: 0,
      fit: 1, shift: 0, drop: 0, last: 0
    };
    let raf = 0;
    let visible = true;
    let active = -1;
    let ready = false;

    const nearest = angle => Math.round(angle / step) * step;

    const setCaption = index => {
      if (!captions) return;
      const item = items[index];
      if (!item || !capTitle) return;
      capTitle.textContent = '';
      capTitle.append(document.createTextNode(item.title || item.alt || ''));
      if (item.subtitle) {
        const sub = document.createElement('span');
        sub.className = 'circular-carousel__subtitle';
        sub.textContent = item.subtitle;
        capTitle.appendChild(sub);
      }
      const pad2 = String(index + 1).padStart(2, '0');
      [...capDigits.children].forEach((digit, i) => {
        digit.firstChild.style.transform = `translateY(${-Number(pad2[i]) * 10}%)`;
      });
      capTitle.style.animation = 'none';
      void capTitle.offsetWidth;
      capTitle.style.animation = '';
    };
    setCaption(0);

    /* ---------- 尺寸适配 ---------- */
    const measure = () => {
      const rect = root.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const room = captions ? CAPTION_SPACE : 0;
      const width = rect.width * 0.94;
      const height = (rect.height - room) * 0.92;
      const P = perspective;
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      if (layout.inward) {
        minX = -width / 2; maxX = width / 2;
        minY = -cardH / 2; maxY = cardH / 2;
      } else {
        const corners = [
          [-cardW / 2, -cardH / 2], [cardW / 2, -cardH / 2],
          [-cardW / 2, cardH / 2], [cardW / 2, cardH / 2]
        ];
        const limit = layout.window ? layout.window * step : 180;
        const rotY = (p, deg) => {
          const r = deg * TO_RAD, c = Math.cos(r), s = Math.sin(r);
          return [p[0] * c + p[2] * s, p[1], -p[0] * s + p[2] * c];
        };
        const rotX = (p, deg) => {
          const r = deg * TO_RAD, c = Math.cos(r), s = Math.sin(r);
          return [p[0], p[1] * c - p[2] * s, p[1] * s + p[2] * c];
        };
        for (let a = -limit; a <= limit; a += limit / 24) {
          for (const [cx, cy] of corners) {
            let p;
            if (axis === 'x') {
              p = rotY(rotX([cx, cy, radius], -a), tiltValue);
            } else if (layout.billboard) {
              const c = rotY([0, 0, radius], a);
              p = rotX([c[0] + cx, cy, c[2]], tiltValue);
            } else {
              p = rotX(rotY([cx, cy, radius], a), tiltValue);
            }
            p = [p[0], p[1], p[2] - radius];
            if (p[2] >= P * 0.95) continue;
            const k = P / (P - p[2]);
            minX = Math.min(minX, p[0] * k);
            maxX = Math.max(maxX, p[0] * k);
            minY = Math.min(minY, p[1] * k);
            maxY = Math.max(maxY, p[1] * k);
          }
        }
      }
      const spanX = Math.max(maxX - minX, 1);
      const spanY = Math.max(maxY - minY, 1);
      state.fit = Math.min(1, width / spanX, height / spanY);
      state.shift = -((minY + maxY) / 2) * state.fit - room / 2;
      state.drop = axis === 'x'
        ? (rect.width / state.fit) * 0.55 + cardW
        : (rect.height / state.fit) * 0.55 + cardH;
      stage.style.perspective = `${P}px`;
      stage.style.transform = `translate3d(0, ${state.shift}px, 0) scale(${state.fit})`;
    };

    /* ---------- 入场 ---------- */
    const introCard = (elapsed, landing) => {
      if (!state.intro) return { radius: 1, lift: 0 };
      const type = state.intro.type;
      const reach = Math.abs(wrap(landing + state.angle));
      if (type === 'assemble') {
        const delay = (reach / 180) * 420;
        const p = easeOut(clamp((elapsed - delay) / 1080, 0, 1));
        return { radius: 1 + 0.6 * (1 - p), lift: 0 };
      }
      if (type === 'rise') {
        const delay = (reach / 180) * 480;
        const p = easeOutQuint(clamp((elapsed - delay) / 900, 0, 1));
        return { radius: 1, lift: (1 - p) * state.drop };
      }
      if (type === 'spin') {
        const p = easeOut(clamp(elapsed / INTRO_LENGTH.spin, 0, 1));
        return { radius: 1 + 0.28 * (1 - p), lift: 0 };
      }
      return { radius: 1, lift: 0 };
    };

    /* ---------- 物理推进 ---------- */
    const advance = (dt, now) => {
      if (!state.introDone && ready) {
        if (!state.intro) {
          const intro = reduced ? 'none' : (o.intro in INTRO_LENGTH ? o.intro : 'rise');
          if (intro === 'none') state.introDone = true;
          else state.intro = { type: intro, start: now };
        }
        if (state.intro && now - state.intro.start >= INTRO_LENGTH[state.intro.type]) {
          state.intro = null;
          state.introDone = true;
        }
      }

      const paused = (o.pauseOnHover && state.hover) || state.drag || now < state.holdUntil;
      const autoplay = reduced ? 'off' : o.autoplay;
      const cruise = autoplay === 'drift' && !paused && !state.intro ? o.speed * state.dir : 0;
      let busy = Boolean(state.intro) || state.drag;

      if (state.drag || state.intro) {
        if (!state.drag) state.velocity = 0;
      } else if (state.target !== null) {
        let remaining = dt;
        const damping = 2 * Math.sqrt(SPRING);
        while (remaining > 0) {
          const h = Math.min(remaining, 1 / 240);
          const accel = SPRING * (state.target - state.angle) - damping * state.velocity;
          state.velocity += accel * h;
          state.angle += state.velocity * h;
          remaining -= h;
        }
        if (Math.abs(state.target - state.angle) < 0.004 && Math.abs(state.velocity) < 0.03) {
          state.angle = state.target;
          state.velocity = 0;
          state.target = null;
        }
        busy = true;
      } else {
        const tau = 0.18 + clamp(o.momentum, 0, 1) * 1.5;
        state.velocity += (cruise - state.velocity) * (1 - Math.exp(-dt / tau));
        state.angle += state.velocity * dt;
        if (cruise === 0 && o.snap && Math.abs(state.velocity) < SETTLE_SPEED) {
          state.target = nearest(state.angle);
        }
        busy = busy || cruise !== 0 || Math.abs(state.velocity) > 0.01 || state.target !== null;
      }

      if (autoplay === 'step' && !paused && !state.intro && state.introDone) {
        if (!state.stepAt) state.stepAt = now + Math.max(0.5, o.interval) * 1000;
        if (now >= state.stepAt) {
          state.target = (state.target ?? nearest(state.angle)) + step * state.dir;
          state.stepAt = now + Math.max(0.5, o.interval) * 1000;
        }
        busy = true;
      } else {
        state.stepAt = 0;
      }

      if (now < state.holdUntil) busy = true;

      const parallax = reduced ? 0 : clamp(o.parallax, 0, 1);
      const ease = 1 - Math.exp(-dt / 0.35);
      const aimYaw = state.pointer.inside ? state.pointer.x * parallax * 9 : 0;
      const aimPitch = state.pointer.inside ? -state.pointer.y * parallax * 6 : 0;
      state.yaw += (aimYaw - state.yaw) * ease;
      state.pitch += (aimPitch - state.pitch) * ease;
      if (Math.abs(aimYaw - state.yaw) > 0.01 || Math.abs(aimPitch - state.pitch) > 0.01) busy = true;

      return busy;
    };

    /* ---------- 渲染 ---------- */
    const render = now => {
      const elapsed = state.intro ? now - state.intro.start : 0;
      const stretch = reduced ? 0 : clamp(o.stretch, 0, 1);
      const swell = 1 + stretch * 0.12 * Math.min(1, Math.abs(state.velocity) / 420);
      let spinOffset = 0;
      if (state.intro?.type === 'spin') {
        const p = easeOut(clamp(elapsed / INTRO_LENGTH.spin, 0, 1));
        spinOffset = -300 * state.dir * (1 - p);
      } else if (state.intro?.type === 'assemble') {
        const p = easeOut(clamp(elapsed / INTRO_LENGTH.assemble, 0, 1));
        spinOffset = -32 * state.dir * (1 - p);
      }
      const angle = state.angle + spinOffset;
      const R = radius * swell;

      if (axis === 'x') {
        camera.style.transform =
          `translate3d(0, 0, ${-R}px) rotateY(${tiltValue + state.yaw}deg) rotateX(${state.pitch}deg)`;
        ring.style.transform = `rotateX(${-angle}deg)`;
      } else if (layout.inward) {
        camera.style.transform =
          `translate3d(0, 0, ${perspective - 1}px) rotateX(${tiltValue + state.pitch}deg) rotateY(${state.yaw}deg)`;
        ring.style.transform = `rotateY(${angle}deg)`;
      } else {
        camera.style.transform =
          `translate3d(0, 0, ${-R}px) rotateX(${tiltValue + state.pitch}deg) rotateY(${state.yaw}deg)`;
        ring.style.transform = `rotateY(${angle}deg)`;
      }

      for (let index = 0; index < count; index++) {
        const card = cardRefs[index];
        const base = index * step;
        const mod = introCard(elapsed, base);
        const r = R * mod.radius;
        let transform;
        if (axis === 'x') {
          transform = `rotateX(${-base}deg) translateZ(${r}px)`;
        } else if (layout.inward) {
          transform = `rotateY(${base}deg) translateZ(${-r}px)`;
        } else {
          transform = `rotateY(${base}deg) translateZ(${r}px)`;
          if (layout.billboard) transform += ` rotateY(${-(base + angle)}deg)`;
        }
        if (mod.lift) transform += axis === 'x'
          ? ` translateX(${mod.lift}px)` : ` translateY(${mod.lift}px)`;
        card.style.transform = transform;

        const world = wrap(base + angle);
        const facing = Math.cos(world * TO_RAD);
        if (layout.inward) card.style.visibility = Math.abs(world) > 86 ? 'hidden' : '';
        const fade = clamp(o.depthFade, 0, 1) * Math.pow((1 - facing) / 2, 1.25);
        card.style.setProperty('--cc-depth', fade.toFixed(3));
      }

      const index = ((Math.round(-state.angle / step) % count) + count) % count || 0;
      if (index !== active) {
        active = index;
        setCaption(index);
        live.textContent = `${items[index]?.title || items[index]?.alt || `Image ${index + 1}`}, ${index + 1} / ${count}`;
      }
    };

    const frame = now => {
      raf = 0;
      const dt = state.last ? Math.min((now - state.last) / 1000, 0.05) : 1 / 60;
      state.last = now;
      const busy = advance(dt, now);
      render(now);
      if (busy && visible && !document.hidden) raf = requestAnimationFrame(frame);
      else state.last = 0;
    };

    const wake = () => {
      if (!raf && visible && !document.hidden) raf = requestAnimationFrame(frame);
    };

    /* ---------- 交互 ---------- */
    const updatePointer = e => {
      const rect = root.getBoundingClientRect();
      state.pointer.x = clamp(((e.clientX - rect.left) / rect.width) * 2 - 1, -1, 1);
      state.pointer.y = clamp(((e.clientY - rect.top) / rect.height) * 2 - 1, -1, 1);
    };

    const onPointerDown = e => {
      state.suppressClick = false;
      if (!o.draggable || e.button !== 0) return;
      state.press = {
        id: e.pointerId, x: e.clientX, y: e.clientY,
        angle: state.angle, moved: false, origin: 0,
        samples: [{ time: performance.now(), angle: state.angle }]
      };
    };

    const onPointerMove = e => {
      if (e.pointerType === 'mouse') {
        state.pointer.inside = true;
        updatePointer(e);
      }
      const press = state.press;
      if (!press || press.id !== e.pointerId) { wake(); return; }
      const delta = axis === 'x' ? e.clientY - press.y : e.clientX - press.x;
      const cross = axis === 'x' ? e.clientX - press.x : e.clientY - press.y;
      if (!press.moved) {
        if (Math.abs(delta) < DRAG_THRESHOLD) return;
        if (Math.abs(cross) > Math.abs(delta) * 1.2 && e.pointerType !== 'mouse') {
          state.press = null;
          return;
        }
        press.moved = true;
        press.origin = delta;
        state.drag = true;
        state.target = null;
        state.velocity = 0;
        root.setAttribute('data-dragging', '');
        try { root.setPointerCapture(e.pointerId); } catch {}
      }
      const perPixel = 180 / (Math.PI * radius * state.fit);
      state.angle = press.angle + (delta - press.origin) * perPixel * dragSign;
      const now = performance.now();
      press.samples.push({ time: now, angle: state.angle });
      while (press.samples.length > 2 && now - press.samples[0].time > 110) press.samples.shift();
      wake();
    };

    const releasePointer = e => {
      const press = state.press;
      if (!press || press.id !== e.pointerId) return;
      state.press = null;
      if (!press.moved) return;
      state.drag = false;
      root.removeAttribute('data-dragging');
      state.suppressClick = true;
      const first = press.samples[0];
      const last = press.samples[press.samples.length - 1];
      const span = (last.time - first.time) / 1000;
      const velocity = span > 0.008 ? clamp((last.angle - first.angle) / span, -1400, 1400) : 0;
      state.velocity = velocity;
      if (Math.abs(velocity) > 60) state.dir = Math.sign(velocity);
      const coasting = o.autoplay === 'drift' && !(o.pauseOnHover && state.hover && e.pointerType === 'mouse');
      if (o.snap && !coasting) {
        const tau = 0.18 + clamp(o.momentum, 0, 1) * 1.5;
        state.target = Math.round((state.angle + velocity * tau * 0.55) / step) * step;
      }
      wake();
    };

    const onPointerEnter = e => {
      if (e.pointerType !== 'mouse') return;
      state.hover = true;
      wake();
    };
    const onPointerLeave = e => {
      if (e.pointerType === 'mouse') {
        state.hover = false;
        state.pointer.inside = false;
      }
      wake();
    };

    const onClick = e => {
      if (state.suppressClick) { state.suppressClick = false; return; }
      const card = e.target.closest?.('[data-cc-index]');
      if (!card) return;
      const index = Number(card.getAttribute('data-cc-index'));
      if (o.focusOnClick) {
        let target = -index * step;
        target += 360 * Math.round((state.angle - target) / 360);
        state.target = target;
        state.holdUntil = performance.now() + 2800;
        wake();
      }
    };

    const stepBy = delta => {
      const base = state.target ?? Math.round(state.angle / step) * step;
      state.target = base - delta * step * dragSign;
      state.holdUntil = performance.now() + 2800;
      wake();
    };

    const onKeyDown = e => {
      const forward = axis === 'x' ? 'ArrowDown' : 'ArrowRight';
      const backward = axis === 'x' ? 'ArrowUp' : 'ArrowLeft';
      if (e.key === forward) stepBy(1);
      else if (e.key === backward) stepBy(-1);
      else return;
      e.preventDefault();
    };

    const onWheel = e => {
      if (!o.draggable) return;
      const delta = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : 0;
      if (!delta) return;
      e.preventDefault();
      const perPixel = 180 / (Math.PI * radius * state.fit);
      const sign = layout.inward ? -1 : 1;
      state.target = null;
      state.angle -= delta * perPixel * sign;
      state.velocity = -delta * perPixel * sign * 30;
      state.holdUntil = performance.now() + 1600;
      clearTimeout(state.wheelTimer);
      state.wheelTimer = setTimeout(() => {
        if (o.snap) state.target = nearest(state.angle + state.velocity * 0.12);
        wake();
      }, 140);
      wake();
    };

    const onVisibility = () => {
      if (document.hidden) {
        cancelAnimationFrame(raf); raf = 0; state.last = 0;
      } else wake();
    };

    root.addEventListener('pointerdown', onPointerDown);
    root.addEventListener('pointermove', onPointerMove);
    root.addEventListener('pointerup', releasePointer);
    root.addEventListener('pointercancel', releasePointer);
    root.addEventListener('pointerenter', onPointerEnter);
    root.addEventListener('pointerleave', onPointerLeave);
    root.addEventListener('click', onClick);
    root.addEventListener('keydown', onKeyDown);
    root.addEventListener('wheel', onWheel, { passive: false });
    document.addEventListener('visibilitychange', onVisibility);

    const resizeObserver = new ResizeObserver(() => { measure(); wake(); });
    resizeObserver.observe(root);
    const io = new IntersectionObserver(entries => {
      visible = entries[0]?.isIntersecting ?? false;
      if (visible) wake();
      else { cancelAnimationFrame(raf); raf = 0; state.last = 0; }
    });
    io.observe(root);

    /* 图片加载完成（或 2.4s 超时）后开始入场 */
    const sources = items.slice(0, 16).map(item => item.src);
    const load = src => new Promise(resolve => {
      const image = new Image();
      image.decoding = 'async';
      image.onload = () => (image.decode ? image.decode().then(resolve, resolve) : resolve());
      image.onerror = resolve;
      image.src = src;
    });
    Promise.race([Promise.all(sources.map(load)), new Promise(r => setTimeout(r, 2400))])
      .then(() => {
        ready = true;
        root.setAttribute('data-ready', '');
        state.introDone = false;
        state.intro = null;
        wake();
      });

    measure();
    render(performance.now());
    wake();

    return {
      destroy() {
        cancelAnimationFrame(raf);
        resizeObserver.disconnect();
        io.disconnect();
        clearTimeout(state.wheelTimer);
        document.removeEventListener('visibilitychange', onVisibility);
      }
    };
  }

  window.TechCarousel = { create };
})();
