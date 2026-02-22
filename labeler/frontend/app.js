// ===== Bilingual translations [English, Thai] =====
const TRANSLATIONS = {
    classes:                ['Classes', 'คลาส'],
    add_class:              ['Add class', 'เพิ่มคลาส'],
    brush_tool:             ['Brush', 'แปรง'],
    eraser_tool:            ['Eraser', 'ยางลบ'],
    run_kmeans:             ['Run K-Means', 'รัน K-Means'],
    reassign_clusters:      ['Re-assign Clusters', 'กำหนดคลัสเตอร์ใหม่'],
    image_adjust:           ['Image Adjust', 'ปรับภาพ'],
    reset:                  ['Reset', 'รีเซ็ต'],
    coverage:               ['Coverage', 'ความครอบคลุม'],
    refresh_stats:          ['Refresh stats', 'รีเฟรชสถิติ'],
    upload_hint:            ['Click or drop images to upload tiles', 'คลิกหรือลากรูปภาพเพื่ออัปโหลด'],
    upload_btn:             ['Upload', 'อัปโหลด'],
    open_sidebar:           ['Open sidebar', 'เปิดแถบด้านซ้าย'],
    open_workflow:          ['Open workflow panel', 'เปิดแผงขั้นตอน'],
    toggle_dark:            ['Toggle dark mode', 'สลับโหมดมืด'],
    switch_lang:            ['Switch language', 'เปลี่ยนภาษา'],
    zoom_in:                ['Zoom in', 'ซูมเข้า'],
    zoom_out:               ['Zoom out', 'ซูมออก'],
    pan_up:                 ['Pan up', 'เลื่อนขึ้น'],
    pan_left:               ['Pan left', 'เลื่อนซ้าย'],
    pan_right:              ['Pan right', 'เลื่อนขวา'],
    pan_down:               ['Pan down', 'เลื่อนลง'],
    reset_view_btn:         ['Fit', 'พอดีจอ'],
    reset_view_title:       ['Reset view', 'รีเซ็ตมุมมอง'],
    sam_hint:               ['SAM preview — Enter accept, Esc reject', 'ตัวอย่าง SAM — Enter ยืนยัน, Esc ยกเลิก'],
    undo:                   ['Undo', 'เลิกทำ'],
    clear:                  ['Clear', 'ล้าง'],
    save:                   ['Save', 'บันทึก'],
    prev_tile:              ['Previous tile', 'ไทล์ก่อนหน้า'],
    next_tile:              ['Next tile', 'ไทล์ถัดไป'],
    unlabeled_only:         ['Unlabeled only', 'ที่ยังไม่ติดป้าย'],
    keyboard_shortcuts:     ['Keyboard shortcuts', 'ปุ่มลัด'],
    delete_tile_btn:        ['Delete image', 'ลบภาพ'],
    delete_tile_title:      ['Delete tile and label', 'ลบไทล์และป้ายกำกับ'],
    workflow_panel:         ['Workflow', 'ขั้นตอน'],
    collapse_panel:         ['Collapse panel', 'ยุบแผง'],
    augment_section:        ['Augment', 'เพิ่มข้อมูล'],
    flip_h:                 ['Flip Horizontal', 'พลิกแนวนอน'],
    flip_v:                 ['Flip Vertical', 'พลิกแนวตั้ง'],
    rotate_aug:             ['Rotate 90° / 180° / 270°', 'หมุน 90° / 180° / 270°'],
    brightness_aug:         ['Brightness / Contrast', 'ความสว่าง / คอนทราสต์'],
    blur_aug:               ['Blur', 'เบลอ'],
    crop_zoom_aug:          ['Crop / Zoom', 'ครอป / ซูม'],
    copies_label:           ['Copies (random aug)', 'จำนวนสำเนา (สุ่ม)'],
    labeled_only:           ['Labeled tiles only', 'เฉพาะไทล์ที่ติดป้าย'],
    run_augmentation:       ['Run Augmentation', 'รันการเพิ่มข้อมูล'],
    processing:             ['Processing...', 'กำลังประมวลผล...'],
    augmented_tiles_label:  ['Augmented tiles', 'ไทล์ที่เพิ่มแล้ว'],
    clear_aug_tiles:        ['Clear Augmented Tiles', 'ลบไทล์ที่เพิ่ม'],
    split_section:          ['Split', 'แบ่งชุดข้อมูล'],
    done_badge:             ['✓ done', '✓ เสร็จ'],
    train_label:            ['Train', 'เทรน'],
    val_label:              ['Val', 'ตรวจสอบ'],
    test_label:             ['Test', 'ทดสอบ'],
    generate_split:         ['Generate Split', 'สร้างการแบ่ง'],
    export_section:         ['Export', 'ส่งออก'],
    export_zip_btn:         ['Export as ZIP', 'ส่งออกเป็น ZIP'],
    requires_splits:        ['Requires splits to be generated first', 'ต้องสร้างการแบ่งก่อน'],
    confirm_title:          ['Confirm', 'ยืนยัน'],
    cancel:                 ['Cancel', 'ยกเลิก'],
    confirm_btn:            ['Confirm', 'ยืนยัน'],
    add_class_title:        ['Add New Class', 'เพิ่มคลาสใหม่'],
    class_name_label:       ['Class name', 'ชื่อคลาส'],
    class_placeholder:      ['e.g. forest', 'เช่น ป่าไม้'],
    color_label:            ['Color', 'สี'],
    add_class_btn:          ['Add Class', 'เพิ่มคลาส'],
    kmeans_title:           ['K-Means Result', 'ผลลัพธ์ K-Means'],
    kmeans_instructions:    ['Assign each cluster to a class:', 'กำหนดแต่ละคลัสเตอร์ให้กับคลาส:'],
    apply_canvas:           ['Apply to Canvas', 'นำไปใช้กับแคนวาส'],
    shortcuts_title:        ['Keyboard Shortcuts', 'ปุ่มลัด'],
    close_btn:              ['Close', 'ปิด'],
    sc_select_class:        ['Select class', 'เลือกคลาส'],
    sc_save:                ['Save', 'บันทึก'],
    sc_undo:                ['Undo', 'เลิกทำ'],
    sc_prevnext:            ['Prev / Next', 'ก่อนหน้า / ถัดไป'],
    sc_sam_point:           ['SAM point', 'จุด SAM'],
    sc_sam_neg:             ['SAM neg point', 'จุด SAM เชิงลบ'],
    sc_sam_box:             ['SAM box', 'กรอบ SAM'],
    sc_accept_sam:          ['Accept SAM', 'ยืนยัน SAM'],
    sc_reject_sam:          ['Reject SAM', 'ยกเลิก SAM'],
    sc_zoom:                ['Zoom', 'ซูม'],
    sc_pan:                 ['Pan', 'เลื่อน'],
    // Dynamic strings (use {placeholder} for variables)
    ready:                  ['Ready', 'พร้อม'],
    switching_folder:       ['Switching folder...', 'กำลังเปลี่ยนโฟลเดอร์...'],
    no_tiles:               ['No tiles in this folder', 'ไม่มีไทล์ในโฟลเดอร์นี้'],
    loading_tiles:          ['Loading tiles...', 'กำลังโหลดไทล์...'],
    loading:                ['Loading...', 'กำลังโหลด...'],
    computing_sam:          ['Computing SAM embedding...', 'กำลังคำนวณ SAM...'],
    running_kmeans:         ['Running K-Means...', 'กำลังรัน K-Means...'],
    kmeans_applied:         ['K-Means applied', 'นำ K-Means ไปใช้แล้ว'],
    saved_status:           ['Saved!', 'บันทึกแล้ว!'],
    no_tiles_remaining:     ['No tiles remaining', 'ไม่มีไทล์เหลือ'],
    running_sam:            ['Running SAM...', 'กำลังรัน SAM...'],
    mask_applied:           ['Mask applied', 'นำมาสก์ไปใช้แล้ว'],
    mask_rejected:          ['Mask rejected', 'ปฏิเสธมาสก์แล้ว'],
    kmeans_failed:          ['K-Means failed: {detail}', 'K-Means ล้มเหลว: {detail}'],
    delete_failed:          ['Failed to delete: {detail}', 'ลบไม่สำเร็จ: {detail}'],
    upload_failed:          ['Upload failed', 'อัปโหลดไม่สำเร็จ'],
    uploaded_tiles:         ['Uploaded {count} tile(s)', 'อัปโหลด {count} ไทล์สำเร็จ'],
    select_aug:             ['Select at least one augmentation', 'เลือกการเพิ่มข้อมูลอย่างน้อยหนึ่งรายการ'],
    segment_min:            ['Each segment must be at least 1%', 'แต่ละส่วนต้องมีอย่างน้อย 1%'],
    split_success:          ['Split: {train} train / {val} val / {test} test', 'แบ่ง: {train} เทรน / {val} ตรวจสอบ / {test} ทดสอบ'],
    split_failed:           ['Split failed: {detail}', 'การแบ่งล้มเหลว: {detail}'],
    removed_aug:            ['Removed {count} augmented tile(s)', 'ลบ {count} ไทล์ที่เพิ่มแล้ว'],
    clear_failed:           ['Clear failed', 'การล้างล้มเหลว'],
    created_aug:            ['Created {count} augmented tiles from {src} source tiles', 'สร้าง {count} ไทล์จาก {src} ไทล์ต้นฉบับ'],
    aug_failed:             ['Augmentation failed', 'การเพิ่มข้อมูลล้มเหลว'],
    aug_error:              ['Augmentation error: {msg}', 'ข้อผิดพลาดการเพิ่มข้อมูล: {msg}'],
    export_failed:          ['Export failed', 'การส่งออกล้มเหลว'],
    export_failed_msg:      ['Export failed: {msg}', 'การส่งออกล้มเหลว: {msg}'],
    dataset_downloaded:     ['Dataset ZIP downloaded', 'ดาวน์โหลด ZIP ชุดข้อมูลสำเร็จ'],
    delete_tile_confirm_title: ['Delete Tile', 'ลบไทล์'],
    delete_tile_confirm_msg:   ['Delete "{name}" and all its associated files?\nThis cannot be undone.',
                                'ลบ "{name}" และไฟล์ที่เกี่ยวข้องทั้งหมด?\nไม่สามารถยกเลิกได้'],
    clear_aug_confirm_title:   ['Clear Augmented Tiles', 'ลบไทล์ที่เพิ่ม'],
    clear_aug_confirm_msg:     ['Delete all augmented tiles (_aug_*) and their labels?\nThis cannot be undone.',
                                'ลบไทล์ที่เพิ่มทั้งหมด (_aug_*) และป้ายกำกับ?\nไม่สามารถยกเลิกได้'],
    labeled_count:          ['{labeled} / {total} labeled', '{labeled} / {total} ติดป้ายแล้ว'],
    augmenting_progress:    ['Augmenting {done} / {total} tiles', 'กำลังเพิ่มข้อมูล {done} / {total} ไทล์'],
    aug_done_label:         ['Done — {count} tiles created', 'เสร็จแล้ว — สร้าง {count} ไทล์'],
    clearing_aug:           ['Clearing old augmentations...', 'กำลังลบการเพิ่มข้อมูลเดิม...'],
    building_zip:           ['Building ZIP… {done}/{total} tiles', 'กำลังสร้าง ZIP… {done}/{total} ไทล์'],
    downloading:            ['Downloading...', 'กำลังดาวน์โหลด...'],
    building_zip_status:    ['Building ZIP...', 'กำลังสร้าง ZIP...'],
    augmenting_status:      ['Augmenting...', 'กำลังเพิ่มข้อมูล...'],
    generating_splits:      ['Generating splits...', 'กำลังสร้างการแบ่ง...'],
    clearing_aug_status:    ['Clearing augmented tiles...', 'กำลังลบไทล์ที่เพิ่ม...'],
    cluster_label:          ['Cluster {index}', 'คลัสเตอร์ {index}'],
    sam_score:              ['SAM score: {score}', 'คะแนน SAM: {score}'],
    server_unreachable:     ['Cannot connect to server — is the backend running?', 'ไม่สามารถเชื่อมต่อเซิร์ฟเวอร์ได้ — ตรวจสอบว่าเซิร์ฟเวอร์กำลังทำงาน'],
    filter_all:             ['All', 'ทั้งหมด'],
    filter_labeled:         ['Labeled', 'ติดป้ายแล้ว'],
    filter_unlabeled:       ['Unlabeled', 'ยังไม่ติดป้าย'],
};

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
    splits: null,       // {train: [...], val: [...], test: [...]} or null
    splitFilter: 'all', // 'all' | 'train' | 'val' | 'test'
    filterMode: 'all',  // 'all' | 'labeled' | 'unlabeled'
    splitH1: 70,        // train/val boundary = train%
    splitH2: 85,        // val/test boundary  = train% + val%
    lang: 'en',         // 'en' | 'th'

    activeTool: 'sam',  // sam, brush, eraser
    isDrawing: false,
    samPoints: [],
    samPointLabels: [],
    samBoxStart: null,
    samHasPreview: false,
    kmeansClusterMask: null,
    kmeansCenters: null,

    // ===== i18n helpers =====
    t(key, params = {}) {
        const pair = TRANSLATIONS[key];
        if (!pair) return key;
        let str = pair[this.lang === 'th' ? 1 : 0];
        for (const [k, v] of Object.entries(params))
            str = str.replaceAll(`{${k}}`, v);
        return str;
    },

    // Returns a friendly string for a caught fetch error
    _fetchErr(e) {
        return e instanceof TypeError ? this.t('server_unreachable') : e.message;
    },

    applyTranslations() {
        document.querySelectorAll('[data-i18n]').forEach(el => {
            el.textContent = this.t(el.dataset.i18n);
        });
        document.querySelectorAll('[data-i18n-title]').forEach(el => {
            el.title = this.t(el.dataset.i18nTitle);
        });
        document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
            el.placeholder = this.t(el.dataset.i18nPlaceholder);
        });
        const langBtn = document.getElementById('btn-lang');
        if (langBtn) langBtn.textContent = this.lang === 'th' ? 'EN' : 'TH';
    },

    async init() {
        Canvas.init();
        await ClassPanel.init();
        await this.loadFolders();
        await this.loadTileList();
        this.setupEventListeners();

        // Restore language preference
        this.lang = localStorage.getItem('lang') || 'en';
        this.applyTranslations();

        if (this.tiles.length > 0) {
            await this.navigateTo(0);
        }
        this.loadStats();
    },

    async loadFolders() {
        const res = await fetch('/api/folders');
        const data = await res.json();
        this.folders = data.folders;
        this.currentFolder = data.current;
    },

    async switchFolder(folder) {
        if (folder === this.currentFolder) return;

        // Save current work before switching
        if (Canvas.isDirty) {
            await this.save();
        }

        this.setStatus(this.t('switching_folder'));
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
                this.setStatus(this.t('no_tiles'));
            }
            this.loadStats();
        }
    },

    async loadTileList() {
        this.setStatus(this.t('loading_tiles'));
        // Two parallel requests — no N serial label checks needed
        const [tilesRes, splitsRes] = await Promise.all([
            fetch('/api/tiles'),
            fetch('/api/splits'),
        ]);
        const data      = await tilesRes.json();
        const splitsData = await splitsRes.json();

        this.tiles = data.tiles;

        // labeled_files comes from the server in one shot — no per-tile fetch
        this.labeledSet.clear();
        for (const f of (data.labeled_files || [])) this.labeledSet.add(f);
        this.updateProgress(data.labeled, data.total);

        this.splits = splitsData.splits || null;
        this.splitFilter = 'all';

        this.applyFilter();
        if (this.updateSplitDisplay) this.updateSplitDisplay();
        this.setStatus(this.t('ready'));
    },


    applyFilter() {
        const splitSet = (this.splits && this.splitFilter !== 'all')
            ? new Set(this.splits[this.splitFilter] || [])
            : null;

        this.filteredTiles = this.tiles.filter(t => {
            if (this.filterMode === 'labeled'   && !this.labeledSet.has(t)) return false;
            if (this.filterMode === 'unlabeled' &&  this.labeledSet.has(t)) return false;
            if (splitSet && !splitSet.has(t)) return false;
            return true;
        });
        this.populateSelect();
        this.renderThumbnails();
    },

    updateFilterCounts() {
        const total     = this.tiles.length;
        const labeled   = this.tiles.filter(t => this.labeledSet.has(t)).length;
        const unlabeled = total - labeled;

        document.getElementById('filter-count-all').textContent       = total;
        document.getElementById('filter-count-labeled').textContent   = labeled;
        document.getElementById('filter-count-unlabeled').textContent = unlabeled;

        ['all', 'labeled', 'unlabeled'].forEach(mode => {
            const btn = document.getElementById(`filter-btn-${mode}`);
            btn.classList.toggle('btn-primary', this.filterMode === mode);
            btn.classList.toggle('btn-ghost',   this.filterMode !== mode);
        });
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
        this.clearKMeansState();

        document.getElementById('tile-select').value = index;
        this.updateThumbnailActive();
        this.setStatus(this.t('loading'));

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
        this.setStatus(this.t('computing_sam'));
        await SAMClient.embed(this.currentFilename);
        this.setStatus(this.t('ready'));
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

    // --- Modal helpers ---

    showConfirm(message, title = 'Confirm') {
        return new Promise((resolve) => {
            const modal = document.getElementById('modal-confirm');
            document.getElementById('modal-confirm-title').textContent = title;
            document.getElementById('modal-confirm-message').textContent = message;

            const okBtn = document.getElementById('btn-confirm-ok');
            const cancelBtn = document.getElementById('btn-confirm-cancel');

            const cleanup = (result) => {
                modal.close();
                okBtn.removeEventListener('click', onOk);
                cancelBtn.removeEventListener('click', onCancel);
                resolve(result);
            };
            const onOk = () => cleanup(true);
            const onCancel = () => cleanup(false);

            okBtn.addEventListener('click', onOk);
            cancelBtn.addEventListener('click', onCancel);
            modal.showModal();
        });
    },

    toast(message, type = 'error') {
        const container = document.getElementById('toast-container');
        const el = document.createElement('div');
        const cls = type === 'success' ? 'alert-success' : type === 'info' ? 'alert-info' : 'alert-error';
        el.className = `alert ${cls} text-sm py-2 px-4 shadow pointer-events-auto`;
        el.textContent = message;
        container.appendChild(el);
        setTimeout(() => el.remove(), 3500);
    },

    async deleteCurrent() {
        if (!this.currentFilename) return;
        const filename = this.currentFilename;

        const confirmed = await this.showConfirm(
            this.t('delete_tile_confirm_msg', { name: filename }),
            this.t('delete_tile_confirm_title')
        );
        if (!confirmed) return;

        const res = await fetch(`/api/tiles/${filename}`, { method: 'DELETE' });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            this.toast(this.t('delete_failed', { detail: err.detail || res.statusText }));
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
            this.setStatus(this.t('no_tiles_remaining'));
        } else {
            const nextIndex = Math.min(filteredIdx, this.filteredTiles.length - 1);
            await this.navigateTo(nextIndex);
        }
    },

    // --- K-Means ---

    async runKMeans() {
        if (!this.currentFilename) return;
        const k = parseInt(document.getElementById('kmeans-k').value);
        const btn = document.getElementById('btn-run-kmeans');
        btn.classList.add('loading');
        btn.disabled = true;
        this.setStatus(this.t('running_kmeans'));
        try {
            const res = await fetch(`/api/kmeans/${this.currentFilename}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ n_clusters: k }),
            });
            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                this.toast(this.t('kmeans_failed', { detail: err.detail || res.statusText }));
                return;
            }
            const data = await res.json();
            this.kmeansClusterMask = await this.decodeLabelMask(data.mask);
            this.kmeansCenters = data.centers;
            this.showKMeansModal(data.n_clusters, data.centers);
        } finally {
            btn.classList.remove('loading');
            btn.disabled = false;
            this.setStatus(this.t('ready'));
        }
    },

    showKMeansModal(nClusters, centers) {
        const list = document.getElementById('kmeans-cluster-list');
        list.innerHTML = '';
        const numClasses = ClassPanel.classes.length;

        for (let i = 0; i < nClusters; i++) {
            const [r, g, b] = centers[i];
            const opts = ClassPanel.classes.map((cls, ci) =>
                `<option value="${ci}" ${ci === i % numClasses ? 'selected' : ''}>${cls.name}</option>`
            ).join('');
            const row = document.createElement('div');
            row.className = 'flex items-center gap-2';
            row.innerHTML = `
                <div class="w-5 h-5 rounded flex-shrink-0 border border-base-300" style="background:rgb(${r},${g},${b})"></div>
                <span class="text-xs text-base-content/70 flex-1">${this.t('cluster_label', { index: i })}</span>
                <select class="select select-bordered select-xs w-28" data-cluster="${i}">${opts}</select>`;
            list.appendChild(row);
        }
        document.getElementById('modal-kmeans').showModal();
    },

    applyKMeans() {
        if (!this.kmeansClusterMask) return;
        const mapping = {};
        document.querySelectorAll('#kmeans-cluster-list select').forEach(sel => {
            mapping[parseInt(sel.dataset.cluster)] = parseInt(sel.value);
        });
        Canvas.pushUndo();
        for (let i = 0; i < Canvas.labelMask.length; i++) {
            Canvas.labelMask[i] = mapping[this.kmeansClusterMask[i]] ?? 0;
        }
        Canvas.renderOverlay();
        document.getElementById('modal-kmeans').close();
        // Keep kmeansClusterMask + kmeansCenters so user can re-assign without re-running
        document.getElementById('btn-reassign-kmeans').classList.remove('hidden');
        this.setStatus(this.t('kmeans_applied'));
    },

    clearKMeansState() {
        this.kmeansClusterMask = null;
        this.kmeansCenters = null;
        document.getElementById('btn-reassign-kmeans').classList.add('hidden');
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
            this.setStatus(this.t('saved_status'));
            this.updateProgressFromServer();
            this.populateSelect();
            this.updateThumbnailLabeled(this.currentIndex);
            this.loadStats();
        }
    },

    async updateProgressFromServer() {
        const res = await fetch('/api/tiles');
        const data = await res.json();
        // Update labeled set in case it changed (e.g. after save)
        for (const f of (data.labeled_files || [])) this.labeledSet.add(f);
        this.updateProgress(data.labeled, data.total);
    },

    async loadStats() {
        try {
            const res = await fetch('/api/stats');
            if (!res.ok) return;
            const data = await res.json();

            document.getElementById('stats-count').textContent =
                this.t('labeled_count', { labeled: data.labeled, total: data.total });

            const barsEl = document.getElementById('stats-bars');
            barsEl.innerHTML = '';
            data.classes.forEach(cls => {
                if (cls.index === 0 && cls.pct === 0) return; // skip empty background
                const [r, g, b] = cls.color;
                const row = document.createElement('div');
                row.className = 'flex items-center gap-1.5';
                row.innerHTML = `
                    <div class="w-2 h-2 rounded-full flex-shrink-0" style="background:rgb(${r},${g},${b})"></div>
                    <span class="text-[10px] text-base-content/60 flex-1 truncate">${cls.name}</span>
                    <div class="w-14 bg-base-200 rounded-full overflow-hidden flex-shrink-0" style="height:5px">
                        <div class="h-full rounded-full" style="width:${cls.pct}%;background:rgb(${r},${g},${b})"></div>
                    </div>
                    <span class="text-[10px] font-mono text-base-content/40 w-7 text-right">${cls.pct}%</span>`;
                barsEl.appendChild(row);
            });
        } catch (_) { /* stats are non-critical */ }
    },

    updateProgress(labeled, total) {
        document.getElementById('progress-text').textContent = `${labeled} / ${total}`;
        const pct = total > 0 ? (labeled / total * 100) : 0;
        const bar = document.getElementById('progress-fill');
        bar.value = pct;
        bar.max = 100;
        this.updateFilterCounts();
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
        this.setStatus(this.t('running_sam'));

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
            this.setStatus(this.t('sam_score', { score: best.score.toFixed(3) }));
        }
    },

    async onSAMBox(x1, y1, x2, y2) {
        this.resetSAMState();
        Canvas.drawBox(x1, y1, x2, y2);
        this.setStatus(this.t('running_sam'));

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
            this.setStatus(this.t('sam_score', { score: best.score.toFixed(3) }));
        }
    },

    acceptSAM() {
        if (Canvas.acceptSAMPreview()) {
            this.resetSAMState();
            this.setStatus(this.t('mask_applied'));
        }
    },

    rejectSAM() {
        this.resetSAMState();
        this.setStatus(this.t('mask_rejected'));
    },

    setActiveTool(tool) {
        this.activeTool = tool;
        document.querySelectorAll('[data-tool]').forEach(b =>
            b.classList.toggle('btn-active', b.dataset.tool === tool));
    },

    toggleLeftPanel() {
        const sidebar = document.getElementById('sidebar-left');
        sidebar.classList.toggle('sidebar-collapsed');
        document.getElementById('sidebar-thumb').classList.toggle('sidebar-collapsed');
        const collapsed = sidebar.classList.contains('sidebar-collapsed');
        document.getElementById('icon-toggle-left').style.transform = collapsed ? 'rotate(180deg)' : '';
        document.getElementById('tab-left').classList.toggle('hidden', !collapsed);
    },

    toggleRightPanel() {
        const sidebar = document.getElementById('sidebar-right');
        sidebar.classList.toggle('sidebar-collapsed');
        const collapsed = sidebar.classList.contains('sidebar-collapsed');
        document.getElementById('icon-toggle-right').style.transform = collapsed ? 'rotate(180deg)' : '';
        document.getElementById('tab-right').classList.toggle('hidden', !collapsed);
    },

    // --- Event listeners ---

    setupEventListeners() {
        const interCanvas = document.getElementById('interaction-canvas');

        // Mouse events on canvas
        interCanvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
        interCanvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
        interCanvas.addEventListener('mouseup', (e) => this.onMouseUp(e));
        interCanvas.addEventListener('contextmenu', (e) => e.preventDefault());

        // Touch events on canvas (single-touch → SAM click / brush paint)
        interCanvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            if (e.touches.length === 1) {
                const t = e.touches[0];
                this.onMouseDown({ clientX: t.clientX, clientY: t.clientY, button: 0 });
            }
        }, { passive: false });

        interCanvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            if (e.touches.length === 1) {
                const t = e.touches[0];
                this.onMouseMove({ clientX: t.clientX, clientY: t.clientY });
            }
        }, { passive: false });

        interCanvas.addEventListener('touchend', (e) => {
            e.preventDefault();
            if (e.changedTouches.length > 0) {
                const t = e.changedTouches[0];
                this.onMouseUp({ clientX: t.clientX, clientY: t.clientY, button: 0 });
            }
        }, { passive: false });

        // Upload tiles (drag-and-drop zone + file picker)
        const uploadZone  = document.getElementById('upload-zone');
        const uploadInput = document.getElementById('upload-input');

        const doUpload = async (files) => {
            if (!files || files.length === 0) return;
            const fd = new FormData();
            for (const f of files) fd.append('files', f);
            this.setStatus(this.t('loading'));
            try {
                const res  = await fetch('/api/upload-tiles', { method: 'POST', body: fd });
                const data = await res.json();
                if (res.ok) {
                    const skippedNote = data.skipped.length ? `, skipped ${data.skipped.length}` : '';
                    this.toast(this.t('uploaded_tiles', { count: data.uploaded.length }) + skippedNote, 'success');
                    await this.loadTileList();
                } else {
                    this.toast(data.detail || this.t('upload_failed'));
                }
            } catch (e) {
                this.toast(this._fetchErr(e));
            } finally {
                this.setStatus(this.t('ready'));
            }
        };

        uploadZone.addEventListener('click', () => uploadInput.click());
        uploadInput.addEventListener('change', (e) => { doUpload(e.target.files); uploadInput.value = ''; });
        uploadZone.addEventListener('dragover', (e) => { e.preventDefault(); uploadZone.classList.add('border-primary'); });
        uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('border-primary'));
        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('border-primary');
            doUpload(e.dataTransfer.files);
        });

        // Sidebar toggle buttons (header chevrons + floating re-open tabs)
        document.getElementById('btn-toggle-left').addEventListener('click', () => this.toggleLeftPanel());
        document.getElementById('btn-toggle-right').addEventListener('click', () => this.toggleRightPanel());
        document.getElementById('tab-left').addEventListener('click', () => this.toggleLeftPanel());
        document.getElementById('tab-right').addEventListener('click', () => this.toggleRightPanel());

        // Dark mode toggle
        document.getElementById('btn-theme').addEventListener('click', () => {
            const html = document.documentElement;
            const isDark = html.getAttribute('data-theme') === 'dark';
            html.setAttribute('data-theme', isDark ? 'cupcake' : 'dark');
            document.getElementById('icon-theme-dark').classList.toggle('hidden', !isDark);
            document.getElementById('icon-theme-light').classList.toggle('hidden', isDark);
            localStorage.setItem('theme', isDark ? 'cupcake' : 'dark');
        });
        // Restore saved theme
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme) {
            document.documentElement.setAttribute('data-theme', savedTheme);
            const isDark = savedTheme === 'dark';
            document.getElementById('icon-theme-dark').classList.toggle('hidden', isDark);
            document.getElementById('icon-theme-light').classList.toggle('hidden', !isDark);
        }

        // Language toggle
        document.getElementById('btn-lang').addEventListener('click', () => {
            this.lang = this.lang === 'en' ? 'th' : 'en';
            localStorage.setItem('lang', this.lang);
            this.applyTranslations();
        });

        // Shortcuts modal
        document.getElementById('btn-shortcuts').addEventListener('click', () =>
            document.getElementById('modal-shortcuts').showModal());

        // Tool tabs (SAM / Brush / K-Means)
        const tabPanels = ['sam', 'brush', 'kmeans'];
        document.querySelectorAll('#tool-tabs .tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('#tool-tabs .tab').forEach(t =>
                    t.classList.toggle('tab-active', t === tab));
                tabPanels.forEach(name => {
                    document.getElementById(`tab-panel-${name}`).classList.toggle('hidden', tab.dataset.tab !== name);
                });
                // Switch active tool to match the selected tab
                if (tab.dataset.tab === 'sam') {
                    this.setActiveTool('sam');
                } else if (tab.dataset.tab === 'brush') {
                    this.setActiveTool('brush');
                }
            });
        });

        // Tool buttons
        document.querySelectorAll('[data-tool]').forEach(btn => {
            btn.addEventListener('click', () => {
                this.setActiveTool(btn.dataset.tool);
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
        ['all', 'labeled', 'unlabeled'].forEach(mode => {
            document.getElementById(`filter-btn-${mode}`).addEventListener('click', () => {
                this.filterMode = mode;
                this.updateFilterCounts();
                this.applyFilter();
                if (this.filteredTiles.length > 0) this.navigateTo(0);
            });
        });




        // Dataset Split — two-handle drag bar
        const splitBar = document.getElementById('split-bar');
        const MIN_SEG  = 2; // minimum % per segment

        this.updateSplitDisplay = () => {
            const train = Math.round(this.splitH1);
            const val   = Math.round(this.splitH2 - this.splitH1);
            const test  = 100 - train - val;

            splitBar.style.background = `linear-gradient(to right,
                #4ade80 ${this.splitH1}%,
                #fbbf24 ${this.splitH1}% ${this.splitH2}%,
                #f87171 ${this.splitH2}% 100%)`;

            document.getElementById('split-handle-1').style.left = this.splitH1 + '%';
            document.getElementById('split-handle-2').style.left = this.splitH2 + '%';

            document.getElementById('split-lbl-train').style.width = train + '%';
            document.getElementById('split-lbl-val').style.width   = val   + '%';

            document.getElementById('split-train-val').textContent = train;
            document.getElementById('split-val-val').textContent   = val;
            document.getElementById('split-test-val').textContent  = test;

            const preview = document.getElementById('split-preview');
            const labeled = [...this.labeledSet];
            const originals = labeled.filter(t => !t.includes('_aug_')).length;
            const augCount  = labeled.filter(t =>  t.includes('_aug_')).length;

            const augInfo = document.getElementById('aug-info');
            if (augCount > 0) {
                document.getElementById('aug-count').textContent = augCount;
                augInfo.classList.remove('hidden');
            } else {
                augInfo.classList.add('hidden');
            }

            if (originals > 0) {
                const nTrainOrig = Math.max(1, Math.round(originals * train / 100));
                const nVal       = Math.max(1, Math.round(originals * val   / 100));
                const nTest      = Math.max(0, originals - nTrainOrig - nVal);
                document.getElementById('split-count-train').textContent = nTrainOrig + augCount;
                document.getElementById('split-count-val').textContent   = nVal;
                document.getElementById('split-count-test').textContent  = nTest;
            } else if (augCount > 0) {
                document.getElementById('split-count-train').textContent = augCount;
                document.getElementById('split-count-val').textContent   = 0;
                document.getElementById('split-count-test').textContent  = 0;
            } else {
                document.getElementById('split-count-train').textContent = '-';
                document.getElementById('split-count-val').textContent   = '-';
                document.getElementById('split-count-test').textContent  = '-';
            }
            preview.classList.add('hidden');
        };

        this.updateSplitDisplay();

        let _splitDrag = null;
        const _splitPct = (clientX) => {
            const rect = splitBar.getBoundingClientRect();
            return Math.max(0, Math.min(100, (clientX - rect.left) / rect.width * 100));
        };
        const _splitMove = (clientX) => {
            if (!_splitDrag) return;
            const pct = _splitPct(clientX);
            if (_splitDrag === 'h1') {
                this.splitH1 = Math.max(MIN_SEG, Math.min(this.splitH2 - MIN_SEG, pct));
            } else {
                this.splitH2 = Math.max(this.splitH1 + MIN_SEG, Math.min(100 - MIN_SEG, pct));
            }
            this.updateSplitDisplay();
        };

        document.getElementById('split-handle-1').addEventListener('mousedown',  (e) => { e.preventDefault(); _splitDrag = 'h1'; });
        document.getElementById('split-handle-2').addEventListener('mousedown',  (e) => { e.preventDefault(); _splitDrag = 'h2'; });
        document.getElementById('split-handle-1').addEventListener('touchstart', (e) => { e.preventDefault(); _splitDrag = 'h1'; }, { passive: false });
        document.getElementById('split-handle-2').addEventListener('touchstart', (e) => { e.preventDefault(); _splitDrag = 'h2'; }, { passive: false });

        window.addEventListener('mousemove', (e) => _splitMove(e.clientX));
        window.addEventListener('mouseup',   ()  => { _splitDrag = null; });
        window.addEventListener('touchmove', (e) => { if (_splitDrag) { e.preventDefault(); _splitMove(e.touches[0].clientX); } }, { passive: false });
        window.addEventListener('touchend',  ()  => { _splitDrag = null; });

        document.getElementById('btn-generate-split').addEventListener('click', async () => {
            const train = Math.round(this.splitH1);
            const val   = Math.round(this.splitH2 - this.splitH1);
            if (train < 1 || val < 1 || train + val >= 100) {
                this.toast(this.t('segment_min'));
                return;
            }
            const btn      = document.getElementById('btn-generate-split');
            const wrap     = document.getElementById('split-progress-wrap');
            const bar      = document.getElementById('split-progress-bar');
            const txt      = document.getElementById('split-progress-text');
            btn.disabled = true;
            bar.value = 0;
            wrap.classList.remove('hidden');
            txt.textContent = this.t('generating_splits');
            this.setStatus(this.t('generating_splits'));

            // Animate bar to 80% while waiting for server response
            let animPct = 0;
            const anim = setInterval(() => {
                animPct = Math.min(animPct + 4, 80);
                bar.value = animPct;
            }, 60);

            try {
                const res = await fetch('/api/splits', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ train_ratio: train / 100, val_ratio: val / 100 }),
                });
                clearInterval(anim);
                const data = await res.json();
                if (res.ok) {
                    bar.value = 100;
                    txt.textContent = this.t('split_success', { train: data.train, val: data.val, test: data.test });
                    this.splits = data.splits;
                    this.splitFilter = 'all';
                    document.getElementById('split-done-badge').classList.remove('hidden');
                    this.applyFilter();
                    await this.loadTileList();
                    this.toast(this.t('split_success', { train: data.train, val: data.val, test: data.test }), 'success');
                    setTimeout(() => wrap.classList.add('hidden'), 2500);
                } else {
                    bar.value = 0;
                    txt.textContent = this.t('split_failed', { detail: data.detail || res.statusText });
                    this.toast(this.t('split_failed', { detail: data.detail || res.statusText }));
                    setTimeout(() => wrap.classList.add('hidden'), 3000);
                }
            } catch (e) {
                clearInterval(anim);
                wrap.classList.add('hidden');
                this.toast(this._fetchErr(e));
            } finally {
                btn.disabled = false;
                this.setStatus(this.t('ready'));
            }
        });

        // Refresh stats button
        document.getElementById('btn-refresh-stats').addEventListener('click', () => this.loadStats());

        // Clear augmented tiles
        document.getElementById('btn-clear-aug').addEventListener('click', async () => {
            const confirmed = await this.showConfirm(
                this.t('clear_aug_confirm_msg'),
                this.t('clear_aug_confirm_title')
            );
            if (!confirmed) return;
            const btn = document.getElementById('btn-clear-aug');
            btn.disabled = true;
            this.setStatus(this.t('clearing_aug_status'));
            try {
                const res = await fetch('/api/augment', { method: 'DELETE' });
                const data = await res.json();
                if (res.ok) {
                    this.toast(this.t('removed_aug', { count: data.deleted_tiles }), 'success');
                    document.getElementById('aug-done-badge').classList.add('hidden');
                    ['aug-flip_h','aug-flip_v','aug-rotate_90','aug-brightness','aug-blur','aug-crop_zoom']
                        .forEach(id => { document.getElementById(id).checked = false; });
                    await this.loadTileList();
                    this.loadStats();
                } else {
                    this.toast(data.detail || this.t('clear_failed'));
                }
            } finally {
                btn.disabled = false;
                this.setStatus(this.t('ready'));
            }
        });

        // Augmentation
        document.getElementById('aug-n').addEventListener('input', (e) => {
            document.getElementById('aug-n-val').textContent = e.target.value;
        });
        // Rotate 90 checkbox controls 90/180/270 together
        document.getElementById('btn-augment').addEventListener('click', async () => {
            const transformMap = {
                'aug-flip_h':    'flip_h',
                'aug-flip_v':    'flip_v',
                'aug-rotate_90': ['rotate_90', 'rotate_180', 'rotate_270'],
                'aug-brightness':'brightness',
                'aug-blur':      'blur',
                'aug-crop_zoom': 'crop_zoom',
            };
            const transforms = [];
            for (const [id, val] of Object.entries(transformMap)) {
                if (document.getElementById(id).checked) {
                    if (Array.isArray(val)) transforms.push(...val);
                    else transforms.push(val);
                }
            }
            if (transforms.length === 0) { this.toast(this.t('select_aug')); return; }
            const btn = document.getElementById('btn-augment');
            const progressEl  = document.getElementById('aug-progress');
            const progressBar = document.getElementById('aug-progress-bar');
            const progressPct = document.getElementById('aug-progress-pct');
            const progressLbl = document.getElementById('aug-progress-label');

            btn.disabled = true;
            progressEl.classList.remove('hidden');
            progressBar.value = 0;
            progressPct.textContent = '0%';
            progressLbl.textContent = this.t('clearing_aug');
            this.setStatus(this.t('augmenting_status'));

            try {
                const res = await fetch('/api/augment', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        transforms,
                        n_random: parseInt(document.getElementById('aug-n').value) || 3,
                        labeled_only: document.getElementById('aug-labeled-only').checked,
                    }),
                });

                const reader  = res.body.getReader();
                const decoder = new TextDecoder();
                let lastData  = null;
                let buffer    = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop(); // keep incomplete line
                    for (const line of lines) {
                        if (!line.startsWith('data: ')) continue;
                        const evt = JSON.parse(line.slice(6));
                        lastData = evt;
                        if (evt.type === 'start') {
                            progressLbl.textContent = this.t('augmenting_progress', { done: 0, total: evt.total });
                        } else if (evt.type === 'progress') {
                            const pct = Math.round(evt.done / evt.total * 100);
                            progressBar.value = pct;
                            progressPct.textContent = `${pct}%`;
                            progressLbl.textContent = this.t('augmenting_progress', { done: evt.done, total: evt.total });
                        } else if (evt.type === 'done') {
                            progressBar.value = 100;
                            progressPct.textContent = '100%';
                            progressLbl.textContent = this.t('aug_done_label', { count: evt.created });
                        }
                    }
                }

                if (lastData?.type === 'done') {
                    const augBadge = document.getElementById('aug-done-badge');
                    augBadge.textContent = `+${lastData.created}`;
                    augBadge.classList.remove('hidden');
                    await this.loadTileList();
                    this.loadStats();
                    this.toast(this.t('created_aug', { count: lastData.created, src: lastData.augmented }), 'success');
                } else {
                    this.toast(this.t('aug_failed'));
                }
            } catch (e) {
                this.toast(this._fetchErr(e));
            } finally {
                btn.disabled = false;
                this.setStatus(this.t('ready'));
                setTimeout(() => progressEl.classList.add('hidden'), 3000);
            }
        });

        // Export dataset as ZIP (two-step: POST→SSE build, GET→download)
        document.getElementById('btn-export-zip').addEventListener('click', async () => {
            const format = document.getElementById('export-zip-format').value;
            const btn = document.getElementById('btn-export-zip');
            const wrap = document.getElementById('export-progress-wrap');
            const bar  = document.getElementById('export-progress-bar');
            const txt  = document.getElementById('export-progress-text');
            btn.disabled = true;
            bar.value = 0;
            wrap.classList.remove('hidden');
            txt.textContent = '0%';
            this.setStatus(this.t('building_zip_status'));
            try {
                // Step 1: POST to build ZIP with SSE progress
                const res = await fetch(`/api/export-zip?format=${format}`, { method: 'POST' });
                if (!res.ok) {
                    const data = await res.json().catch(() => ({}));
                    this.toast(data.detail || this.t('export_failed'));
                    return;
                }
                const reader = res.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                let done = false;
                while (!done) {
                    const { done: streamDone, value } = await reader.read();
                    if (streamDone) break;
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop();
                    for (const line of lines) {
                        if (!line.startsWith('data: ')) continue;
                        const evt = JSON.parse(line.slice(6));
                        if (evt.type === 'progress') {
                            const pct = Math.round(evt.done / evt.total * 100);
                            bar.value = pct;
                            txt.textContent = `${pct}%`;
                            this.setStatus(this.t('building_zip', { done: evt.done, total: evt.total }));
                        } else if (evt.type === 'done') {
                            bar.value = 100;
                            txt.textContent = '100%';
                            done = true;
                        } else if (evt.type === 'error') {
                            throw new Error(evt.message || this.t('export_failed'));
                        }
                    }
                }
                // Step 2: GET to download the pre-built ZIP
                this.setStatus(this.t('downloading'));
                const a = document.createElement('a');
                a.href = `/api/export-zip/download?format=${format}`;
                a.download = `dataset_${format}.zip`;
                a.click();
                this.toast(this.t('dataset_downloaded'), 'success');
                setTimeout(() => { wrap.classList.add('hidden'); this.setStatus(this.t('ready')); }, 2000);
            } catch (e) {
                this.toast(this._fetchErr(e));
                wrap.classList.add('hidden');
                this.setStatus(this.t('ready'));
            } finally {
                btn.disabled = false;
            }
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

        // K-Means
        document.getElementById('kmeans-k').addEventListener('input', (e) => {
            document.getElementById('kmeans-k-value').textContent = e.target.value;
        });
        document.getElementById('btn-run-kmeans').addEventListener('click', () => this.runKMeans());
        document.getElementById('btn-reassign-kmeans').addEventListener('click', () => {
            if (this.kmeansClusterMask && this.kmeansCenters) {
                this.showKMeansModal(this.kmeansCenters.length, this.kmeansCenters);
            }
        });
        document.getElementById('btn-kmeans-apply').addEventListener('click', () => this.applyKMeans());
        document.getElementById('btn-kmeans-cancel').addEventListener('click', () => {
            document.getElementById('modal-kmeans').close();
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
