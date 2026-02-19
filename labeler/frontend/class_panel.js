/**
 * Class Panel - manages class selection UI with pixel counts and class management
 */
const ClassPanel = {
    classes: [],
    activeIndex: 1, // default to first non-background class

    async init() {
        const res = await fetch('/api/config');
        const data = await res.json();
        this.classes = data.classes;
        this.render();
        this.setupManagement();
    },

    render() {
        const container = document.getElementById('class-list');
        container.innerHTML = '';

        this.classes.forEach((cls, i) => {
            const btn = document.createElement('button');
            btn.className = 'class-btn' + (i === this.activeIndex ? ' active' : '');
            btn.dataset.index = i;

            const [r, g, b] = cls.color;
            btn.innerHTML = `
                <span class="class-color" style="background: rgb(${r},${g},${b})"></span>
                <span class="class-name">${cls.name}</span>
                <span class="class-count" id="class-count-${i}">0</span>
                <span class="class-key">${i + 1}</span>
                ${i > 0 ? '<span class="class-delete" title="Delete class">&times;</span>' : ''}
            `;

            btn.addEventListener('click', (e) => {
                if (e.target.classList.contains('class-delete')) {
                    e.stopPropagation();
                    this.removeClass(i);
                    return;
                }
                this.select(i);
            });
            container.appendChild(btn);
        });
    },

    select(index) {
        this.activeIndex = index;
        document.querySelectorAll('.class-btn').forEach(btn => {
            btn.classList.toggle('active', parseInt(btn.dataset.index) === index);
        });
    },

    getColor(index) {
        if (index < this.classes.length) {
            return this.classes[index].color;
        }
        return [0, 0, 0];
    },

    updateCounts() {
        if (!Canvas.labelMask) return;
        const counts = Canvas.getClassCounts();
        const total = 512 * 512;

        this.classes.forEach((cls, i) => {
            const el = document.getElementById(`class-count-${i}`);
            if (!el) return;
            const count = counts[i] || 0;
            const pct = (count / total * 100).toFixed(1);
            el.textContent = pct + '%';
        });
    },

    // --- Class management ---

    setupManagement() {
        const addBtn = document.getElementById('btn-add-class');
        const modal = document.getElementById('modal-add-class');
        const confirmBtn = document.getElementById('btn-confirm-class');
        const cancelBtn = document.getElementById('btn-cancel-class');
        const nameInput = document.getElementById('new-class-name');

        addBtn.addEventListener('click', () => {
            nameInput.value = '';
            modal.showModal();
            setTimeout(() => nameInput.focus(), 50);
        });

        cancelBtn.addEventListener('click', () => modal.close());

        confirmBtn.addEventListener('click', () => this.addClass());

        nameInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') this.addClass();
        });
    },

    async addClass() {
        const nameInput = document.getElementById('new-class-name');
        const colorInput = document.getElementById('new-class-color');
        const modal = document.getElementById('modal-add-class');

        const name = nameInput.value.trim();
        if (!name) {
            nameInput.focus();
            return;
        }

        // Convert hex color to RGB array
        const hex = colorInput.value;
        const r = parseInt(hex.slice(1, 3), 16);
        const g = parseInt(hex.slice(3, 5), 16);
        const b = parseInt(hex.slice(5, 7), 16);

        // Build updated class list
        const updatedClasses = this.classes.map(c => ({ name: c.name, color: c.color }));
        updatedClasses.push({ name, color: [r, g, b] });

        const res = await fetch('/api/classes', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ classes: updatedClasses }),
        });

        if (res.ok) {
            const data = await res.json();
            this.classes = data.classes;
            this.render();
            modal.close();
        } else {
            const err = await res.json().catch(() => ({}));
            alert(`Failed to add class: ${err.detail || res.statusText}`);
        }
    },

    async removeClass(index) {
        const cls = this.classes[index];
        if (!confirm(`Delete class "${cls.name}"?`)) return;

        const res = await fetch(`/api/classes/${index}`, { method: 'DELETE' });

        if (res.ok) {
            const data = await res.json();
            this.classes = data.classes;
            if (this.activeIndex >= this.classes.length) {
                this.activeIndex = this.classes.length - 1;
            }
            this.render();
        } else {
            const err = await res.json().catch(() => ({}));
            alert(`Failed to delete class: ${err.detail || res.statusText}`);
        }
    },
};
