import {
    canvas as paper,
    ui
} from "./scope";

class Cursor {
    constructor() {
        this.cursor = null;

        this.visible = false;
        this.position = new paper.Point(0, 0);
    }

    init() {
        ui.view.on({
            mousemove: event => {
                if (!this.cursor)
                    this.position = event.point;
            },
            mouseleave: event => {
                this.hide();
            },
            mouseenter: event => {
                this.show();
            },
        });

        paper.view.on({
            mousemove: event => {
                this.position = paper.view.matrix.transform(event.point);

                if (this.cursor) {
                    this.show();
                    this.cursor.position = this.position;
                }
            },
            mouseleave: event => {
                this.hide();
            },
            mouseenter: event => {
                this.show();
            },
            click: event => {
                if (event.event.button !== 0 && this.cursor)
                    this.hide();
            }
        });
    }

    activate({
        cursor,
    }) {
        this.remove();

        if (cursor)
            this.cursor = cursor;

        this.cursor.visible = this.visible;
        this.cursor.position = this.position;
    }

    deactivate() {
        this.remove();
        this.cursor = null;
    }

    show() {
        this.visible = true;

        if (this.cursor && this.cursor.visible == false) {
            this.cursor.visible = true;
            paper.view.update();
        }
    }

    hide() {
        this.visible = false;

        if (this.cursor && this.cursor.visible == true) {
            this.cursor.visible = false;
            paper.view.update();
        }
    }

    remove() {
        if (this.cursor)
            this.cursor.remove();
    }
}

export default new Cursor();