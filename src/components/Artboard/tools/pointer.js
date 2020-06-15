import {
    canvas as paper,
    cursor,
} from "../utils";

const pointer = new paper.Tool();

pointer.onActivate = pointer.onDeactivate = event => {
    cursor.deactivate();
};

export default pointer;