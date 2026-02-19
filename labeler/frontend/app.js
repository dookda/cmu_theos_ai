/**
 * App - main application logic
 */
const App = {
    folders: [],
    currentFolder: '',
    tiles: [],
    filteredTiles: [],
    currentIndex: 0,
    currentFilename: null,
    labeledSet: new Set(),

    activeTool: 'sam',  // sam, brush, eraser
    isDrawing: false,
    samPoints: [],
    samPointLabels: [],
    samBoxStart: null,
    samHasPreview: false,

    async init() {
        Canvas.init();
        await ClassPanel.init();
        await this.loadFolders();
        await this.loadExportFormats();
        await this.loadTileList();
        this.setupEventListeners();

        if (this.tiles.length > 0) {
            await this.navigateTo(0);
        }
    },

    async loadFolders() {
        const res = await fetch('/api/folders');
        const data = await res.json();
        this.folders = data.folders;
        this.currentFolder = data.current;

        const select = document.getElementById('folder-select');
        const wrapper = select.closest('.flex');
        select.innerHTML = '';
        this.folders.forEach(folder => {
            const opt = document.createElement('option');
            opt.value = folder;
            opt.textContent = folder || '(all tiles)';
            select.appendChild(opt);
        });
        select.value = this.currentFolder;

        // Hide folder selector when only one folder
        if (wrapper) {
            wrapper.style.display = this.folders.length > 1 ? '' : 'none';
        }
    },

    async switchFolder(folder) {
        if (folder === this.currentFolder) return;

        // Save current work before switching
        if (Canvas.isDirty) {
            await this.save();
        }

        this.setStatus('Switching folder...');
        const res = await fetch('/api/folders/select', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ folder }),
        });
        const data = await res.json();
        if (data.status === 'ok') {
            this.currentFolder = data.current;
            this.currentIndex = 0;
            this.currentFilename = null;
            await this.loadTileList();
            if (this.filteredTiles.length > 0) {
                await this.navigateTo(0);
            } else {
                Canvas.loadMask(new Uint8Array(512 * 512));
                this.setStatus('No tiles in this folder');
            }
        }
    },

    async loadExportFormats() {
        const res = await fetch('/api/export-formats');
        const data = await res.json();
        const formats = new Set(data.formats);
        const detectCb = document.getElementById('fmt-detect');
        const segmentCb = document.getElementById('fmt-segment');
        if (detectCb) detectCb.checked = formats.has('detect');
        if (segmentCb) segmentCb.checked = formats.has('segment');
    },

    async updateExportFormats() {
        const formats = ['semantic'];
        if (document.getElementById('fmt-detect')?.checked) formats.push('detect');
        if (document.getElementById('fmt-segment')?.checked) formats.push('segment');
        await fetch('/api/export-formats', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ formats }),
        });
    },

    async loadTileList() {
        const res = await fetch('/api/tiles');
        const data = await res.json();
        this.tiles = data.tiles;

        // Build labeled set
        const progressRes = await fetch('/api/progress');
        const progressData = await progressRes.json();
        this.updateProgress(progressData.labeled, progressData.total);

        // Check which tiles are labeled
        this.labeledSet.clear();
        for (const t of this.tiles) {
            const labelRes = await fetch(`/api/labels/${t}`);
            const labelData = await labelRes.json();
            if (labelData.exists) this.labeledSet.add(t);
        }

        this.applyFilter();
    },

    applyFilter() {
        const unlabeledOnly = document.getElementById('filter-unlabeled').checked;
        if (unlabeledOnly) {
            this.filteredTiles = this.tiles.filter(t => !this.labeledSet.has(t));
        } else {
            this.filteredTiles = [...this.tiles];
        }
        this.populateSelect();
        this.renderThumbnails();
    },

    renderThumbnails() {
        const container = document.getElementById('thumbnail-list');
        container.innerHTML = '';

        this.filteredTiles.forEach((tile, i) => {
            const div = document.createElement('div');
            div.className = 'thumb-item' + (i === this.currentIndex ? ' active' : '');
            if (this.labeledSet.has(tile)) div.classList.add('labeled');

            const img = document.createElement('img');
            img.src = `/api/tiles/${tile}`;
            img.alt = tile;
            img.loading = 'lazy';

            const label = document.createElement('div');
            label.className = 'thumb-label';
            label.textContent = tile.replace('.png', '').replace('tile_', '');

            div.appendChild(img);
            div.appendChild(label);

            if (this.labeledSet.has(tile)) {
                const badge = document.createElement('div');
                badge.className = 'thumb-badge';
                div.appendChild(badge);

                // Render label overlay on thumbnail
                this.renderThumbOverlay(div, tile);
            }

            div.addEventListener('click', () => this.navigateTo(i));
            container.appendChild(div);
        });
    },

    async renderThumbOverlay(thumbDiv, tile) {
        const res = await fetch(`/api/labels/${tile}`);
        const data = await res.json();
        if (!data.exists || !data.mask) return;

        const maskArray = await this.decodeLabelMask(data.mask);

        const canvas = document.createElement('canvas');
        canvas.className = 'thumb-overlay';
        canvas.width = 512;
        canvas.height = 512;
        const ctx = canvas.getContext('2d');
        const imgData = ctx.createImageData(512, 512);

        for (let i = 0; i < 512 * 512; i++) {
            const cls = maskArray[i];
            if (cls === 0) continue;
            const color = ClassPanel.getColor(cls);
            imgData.data[i * 4] = color[0];
            imgData.data[i * 4 + 1] = color[1];
            imgData.data[i * 4 + 2] = color[2];
            imgData.data[i * 4 + 3] = 140;
        }

        ctx.putImageData(imgData, 0, 0);
        thumbDiv.appendChild(canvas);
    },

    updateThumbnailActive() {
        const items = document.querySelectorAll('.thumb-item');
        items.forEach((item, i) => {
            item.classList.toggle('active', i === this.currentIndex);
        });
        // Scroll active thumbnail into view
        const active = document.querySelector('.thumb-item.active');
        if (active) {
            active.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        }
    },

    updateThumbnailLabeled(index) {
        const items = document.querySelectorAll('.thumb-item');
        if (!items[index]) return;
        const tile = this.filteredTiles[index];

        if (!items[index].classList.contains('labeled')) {
            items[index].classList.add('labeled');
            const badge = document.createElement('div');
            badge.className = 'thumb-badge';
            items[index].appendChild(badge);
        }

        // Remove old overlay and re-render
        const oldOverlay = items[index].querySelector('.thumb-overlay');
        if (oldOverlay) oldOverlay.remove();
        this.renderThumbOverlay(items[index], tile);
    },

    populateSelect() {
        const select = document.getElementById('tile-select');
        select.innerHTML = '';
        this.filteredTiles.forEach((tile, i) => {
            const opt = document.createElement('option');
            opt.value = i;
            const labeled = this.labeledSet.has(tile) ? ' [labeled]' : '';
            opt.textContent = tile + labeled;
            select.appendChild(opt);
        });
        select.value = this.currentIndex;
    },

    async navigateTo(index) {
        if (index < 0 || index >= this.filteredTiles.length) return;

        // Prompt save if dirty
        if (Canvas.isDirty) {
            await this.save();
        }

        this.currentIndex = index;
        this.currentFilename = this.filteredTiles[index];
        this.resetSAMState();

        document.getElementById('tile-select').value = index;
        this.updateThumbnailActive();
        this.setStatus('Loading...');

        // Load image
        await Canvas.loadImage(this.currentFilename);

        // Load existing label
        const res = await fetch(`/api/labels/${this.currentFilename}`);
        const data = await res.json();
        if (data.exists) {
            const maskArray = await this.decodeLabelMask(data.mask);
            Canvas.loadMask(maskArray);
        } else {
            Canvas.loadMask(new Uint8Array(512 * 512));
        }

        // Precompute SAM embedding
        this.setStatus('Computing SAM embedding...');
        await SAMClient.embed(this.currentFilename);
        this.setStatus('Ready');
    },

    async decodeLabelMask(b64) {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement('canvas');
                canvas.width = 512;
                canvas.height = 512;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0);
                const data = ctx.getImageData(0, 0, 512, 512).data;
                const mask = new Uint8Array(512 * 512);
                for (let i = 0; i < 512 * 512; i++) {
                    mask[i] = data[i * 4]; // R channel = class index
                }
                resolve(mask);
            };
            img.src = `data:image/png;base64,${b64}`;
        });
    },

    async deleteCurrent() {
        if (!this.currentFilename) return;
        const filename = this.currentFilename;

        if (!confirm(`Delete "${filename}" and all its associated files?\nThis cannot be undone.`)) return;

        const res = await fetch(`/api/tiles/${filename}`, { method: 'DELETE' });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            alert(`Failed to delete: ${err.detail || res.statusText}`);
            return;
        }

        // Remove from tile lists
        const tileIdx = this.tiles.indexOf(filename);
        if (tileIdx !== -1) this.tiles.splice(tileIdx, 1);
        const filteredIdx = this.filteredTiles.indexOf(filename);
        if (filteredIdx !== -1) this.filteredTiles.splice(filteredIdx, 1);
        this.labeledSet.delete(filename);

        // Navigate to same position (clamped) or clear canvas if no tiles left
        this.currentFilename = null;
        Canvas.isDirty = false;
        this.populateSelect();
        this.renderThumbnails();
        await this.updateProgressFromServer();

        if (this.filteredTiles.length === 0) {
            Canvas.loadMask(new Uint8Array(512 * 512));
            this.setStatus('No tiles remaining');
        } else {
            const nextIndex = Math.min(filteredIdx, this.filteredTiles.length - 1);
            await this.navigateTo(nextIndex);
        }
    },

    async save() {
        if (!this.currentFilename) return;
        const b64 = Canvas.getMaskAsBase64();
        const res = await fetch(`/api/labels/${this.currentFilename}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mask: b64 }),
        });
        const data = await res.json();
        if (data.status === 'saved') {
            Canvas.isDirty = false;
            this.labeledSet.add(this.currentFilename);
            this.setStatus('Saved!');
            this.updateProgressFromServer();
            this.populateSelect();
            this.updateThumbnailLabeled(this.currentIndex);
        }
    },

    async updateProgressFromServer() {
        const res = await fetch('/api/progress');
        const data = await res.json();
        this.updateProgress(data.labeled, data.total);
    },

    updateProgress(labeled, total) {
        document.getElementById('progress-text').textContent = `${labeled} / ${total}`;
        const pct = total > 0 ? (labeled / total * 100) : 0;
        const bar = document.getElementById('progress-fill');
        bar.value = pct;
        bar.max = 100;
    },

    setStatus(msg) {
        document.getElementById('status-msg').textContent = msg;
    },

    resetSAMState() {
        this.samPoints = [];
        this.samPointLabels = [];
        this.samBoxStart = null;
        this.samHasPreview = false;
        Canvas.clearInteraction();
        document.getElementById('sam-hint').classList.add('hidden');
    },

    // --- SAM interaction ---

    async onSAMClick(x, y, isPositive) {
        this.samPoints.push([x, y]);
        this.samPointLabels.push(isPositive ? 1 : 0);

        Canvas.drawPoint(x, y, isPositive);
        this.setStatus('Running SAM...');

        const result = await SAMClient.predict(
            this.currentFilename,
            this.samPoints,
            this.samPointLabels,
            null
        );

        if (result.masks && result.masks.length > 0) {
            const best = result.masks[0];
            const binaryMask = await SAMClient.decodeMaskBase64(best.mask);
            Canvas.renderSAMPreview(binaryMask);
            this.samHasPreview = true;
            document.getElementById('sam-hint').classList.remove('hidden');
            this.setStatus(`SAM score: ${best.score.toFixed(3)}`);
        }
    },

    async onSAMBox(x1, y1, x2, y2) {
        this.resetSAMState();
        Canvas.drawBox(x1, y1, x2, y2);
        this.setStatus('Running SAM...');

        const result = await SAMClient.predict(
            this.currentFilename,
            null,
            null,
            [x1, y1, x2, y2]
        );

        if (result.masks && result.masks.length > 0) {
            const best = result.masks[0];
            const binaryMask = await SAMClient.decodeMaskBase64(best.mask);
            Canvas.renderSAMPreview(binaryMask);
            this.samHasPreview = true;
            document.getElementById('sam-hint').classList.remove('hidden');
            this.setStatus(`SAM score: ${best.score.toFixed(3)}`);
        }
    },

    acceptSAM() {
        if (Canvas.acceptSAMPreview()) {
            this.resetSAMState();
            this.setStatus('Mask applied');
        }
    },

    rejectSAM() {
        this.resetSAMState();
        this.setStatus('Mask rejected');
    },

    // --- Event listeners ---

    setupEventListeners() {
        const interCanvas = document.getElementById('interaction-canvas');

        // Mouse events on canvas
        interCanvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
        interCanvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
        interCanvas.addEventListener('mouseup', (e) => this.onMouseUp(e));
        interCanvas.addEventListener('contextmenu', (e) => e.preventDefault());

        // Tool buttons
        document.querySelectorAll('[data-tool]').forEach(btn => {
            btn.addEventListener('click', () => {
                this.activeTool = btn.dataset.tool;
                document.querySelectorAll('[data-tool]').forEach(b =>
                    b.classList.toggle('btn-active', b === btn));
                this.resetSAMState();
            });
        });

        // Action buttons
        document.getElementById('btn-undo').addEventListener('click', () => Canvas.undo());
        document.getElementById('btn-clear').addEventListener('click', () => Canvas.clearMask());
        document.getElementById('btn-save').addEventListener('click', () => this.save());
        document.getElementById('btn-delete-tile').addEventListener('click', () => this.deleteCurrent());

        // Navigation
        document.getElementById('btn-prev').addEventListener('click', () =>
            this.navigateTo(this.currentIndex - 1));
        document.getElementById('btn-next').addEventListener('click', () =>
            this.navigateTo(this.currentIndex + 1));
        document.getElementById('tile-select').addEventListener('change', (e) =>
            this.navigateTo(parseInt(e.target.value)));
        document.getElementById('filter-unlabeled').addEventListener('change', () =>
            this.applyFilter());

        // Folder selection
        document.getElementById('folder-select').addEventListener('change', (e) =>
            this.switchFolder(e.target.value));

        // Export format checkboxes
        document.querySelectorAll('[data-format]').forEach(cb => {
            cb.addEventListener('change', () => this.updateExportFormats());
        });

        // Brush size
        document.getElementById('brush-size').addEventListener('input', (e) => {
            Canvas.brushSize = parseInt(e.target.value);
            document.getElementById('brush-size-value').textContent = e.target.value;
        });

        // Opacity
        document.getElementById('opacity-slider').addEventListener('input', (e) => {
            const val = parseInt(e.target.value);
            document.getElementById('opacity-value').textContent = val;
            Canvas.setOpacity(val / 100);
        });

        // SAM Grow/Shrink
        document.getElementById('grow-slider').addEventListener('input', (e) => {
            const val = parseInt(e.target.value);
            document.getElementById('grow-value').textContent = val;
            Canvas.setGrowRadius(val);
        });

        // Brightness
        document.getElementById('brightness-slider').addEventListener('input', (e) => {
            const val = parseInt(e.target.value);
            document.getElementById('brightness-value').textContent = val;
            Canvas.setBrightness(val);
        });

        // Contrast
        document.getElementById('contrast-slider').addEventListener('input', (e) => {
            const val = parseInt(e.target.value);
            document.getElementById('contrast-value').textContent = val;
            Canvas.setContrast(val);
        });

        // Reset adjustments
        document.getElementById('btn-reset-adjust').addEventListener('click', () => {
            document.getElementById('brightness-slider').value = 100;
            document.getElementById('brightness-value').textContent = '100';
            document.getElementById('contrast-slider').value = 100;
            document.getElementById('contrast-value').textContent = '100';
            Canvas.setBrightness(100);
            Canvas.setContrast(100);
        });

        // View controls
        const panStep = 50;
        document.getElementById('btn-zoom-in').addEventListener('click', () => {
            Canvas.zoom = Math.min(5, Canvas.zoom + 0.25);
            Canvas.applyTransform();
        });
        document.getElementById('btn-zoom-out').addEventListener('click', () => {
            Canvas.zoom = Math.max(0.5, Canvas.zoom - 0.25);
            Canvas.applyTransform();
        });
        document.getElementById('btn-pan-up').addEventListener('click', () => {
            Canvas.panY += panStep;
            Canvas.applyTransform();
        });
        document.getElementById('btn-pan-down').addEventListener('click', () => {
            Canvas.panY -= panStep;
            Canvas.applyTransform();
        });
        document.getElementById('btn-pan-left').addEventListener('click', () => {
            Canvas.panX += panStep;
            Canvas.applyTransform();
        });
        document.getElementById('btn-pan-right').addEventListener('click', () => {
            Canvas.panX -= panStep;
            Canvas.applyTransform();
        });
        document.getElementById('btn-reset-view').addEventListener('click', () => {
            Canvas.resetZoom();
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.onKeyDown(e));
    },

    getCanvasCoords(e) {
        return Canvas.screenToCanvas(e.clientX, e.clientY);
    },

    onMouseDown(e) {
        const { x, y } = this.getCanvasCoords(e);

        if (this.activeTool === 'sam') {
            if (e.button === 2) {
                this.onSAMClick(x, y, false);
            } else if (e.button === 0) {
                this.samBoxStart = { x, y };
            }
        } else if ((this.activeTool === 'brush' || this.activeTool === 'eraser') && e.button === 0) {
            Canvas.pushUndo();
            this.isDrawing = true;
            const cls = this.activeTool === 'eraser' ? 0 : ClassPanel.activeIndex;
            Canvas.paintAt(x, y, cls);
            Canvas.renderOverlay();
        }
    },

    onMouseMove(e) {
        const { x, y } = this.getCanvasCoords(e);

        if (this.isDrawing && (this.activeTool === 'brush' || this.activeTool === 'eraser')) {
            const cls = this.activeTool === 'eraser' ? 0 : ClassPanel.activeIndex;
            Canvas.paintAt(x, y, cls);
            Canvas.renderOverlay();
        }
    },

    onMouseUp(e) {
        const { x, y } = this.getCanvasCoords(e);

        if (this.activeTool === 'sam' && e.button === 0 && this.samBoxStart) {
            const dx = Math.abs(x - this.samBoxStart.x);
            const dy = Math.abs(y - this.samBoxStart.y);

            if (dx < 5 && dy < 5) {
                this.onSAMClick(x, y, true);
            } else {
                const x1 = Math.min(this.samBoxStart.x, x);
                const y1 = Math.min(this.samBoxStart.y, y);
                const x2 = Math.max(this.samBoxStart.x, x);
                const y2 = Math.max(this.samBoxStart.y, y);
                this.onSAMBox(x1, y1, x2, y2);
            }
            this.samBoxStart = null;
        }

        if (this.isDrawing) {
            this.isDrawing = false;
        }
    },

    onKeyDown(e) {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

        // Number keys 1-7 for class selection
        if (e.key >= '1' && e.key <= '7') {
            ClassPanel.select(parseInt(e.key) - 1);
            return;
        }

        switch (e.key.toLowerCase()) {
            case 's':
                e.preventDefault();
                this.save();
                break;
            case 'z':
                Canvas.undo();
                break;
            case 'enter':
                if (this.samHasPreview) {
                    this.acceptSAM();
                }
                break;
            case 'escape':
                if (this.samHasPreview) {
                    this.rejectSAM();
                }
                break;
            case 'arrowleft':
                e.preventDefault();
                this.navigateTo(this.currentIndex - 1);
                break;
            case 'arrowright':
                e.preventDefault();
                this.navigateTo(this.currentIndex + 1);
                break;
        }
    },
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => App.init());
