class History {
    constructor(capacity) {
        this.capacity = capacity;
        this.init();
    }

    init() {
        this.history = [];
        this.current = 0;
    }

    add(action) {
        // clean tail record
        this.history.splice(this.current, this.history.length);

        // ensure capacity
        if (this.history.length >= this.capacity)
            this.history.shift();

        this.history.push(action);
        this.current = this.history.length;
    }

    undo() {
        if (this.canUndo())
            this.history[--this.current].unexec();
    }

    redo() {
        if (this.canRedo())
            this.history[this.current++].exec();
    }

    canUndo() {
        return this.current > 0;
    }

    canRedo() {
        return this.history.length > this.current;
    }
}

export default new History(20);