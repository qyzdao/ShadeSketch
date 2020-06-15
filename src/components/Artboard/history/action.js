import {
    canvas as paper
} from "../utils";

// command
class CanvasAction {
    constructor(args) {
        this.args = args;
    }
    exec() {
        this.args.item.addTo(paper.project);
    }
    unexec() {
        this.args.item.remove();
    }
}

class ViewAction {
    constructor(args) {
        this.args = args;
    }
    _transform(args, invert = false) {
        for (let attr in args) {
            if (invert)
                paper.project.view[attr] = args[attr].prev
            else
                paper.project.view[attr] = args[attr].next;
        }
    }
    exec() {
        this._transform(this.args);
    }
    unexec() {
        this._transform(this.args, true);
    }
}

export {
    CanvasAction,
    ViewAction,
};