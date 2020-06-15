import {
    canvas as paper,
    cursor,
    option,
    ui,
} from "../utils";
import history, {
    CanvasAction
} from "../history";

const state = {
    path: null,
};

const eraser = new paper.Tool();

eraser.onMouseDown = event => {
    state.path = new paper.Path();
    state.path.strokeColor = "#ffffff";
    state.path.strokeWidth = option.eraser.size / paper.project.view.zoom;
    state.path.strokeCap = "round";
    state.path.strokeJoin = "round";
    // state.path.blendMode = "destination-out";

    state.path.add(event.point);
};

eraser.onMouseDrag = event => {
    if (!state.path) return;

    state.path.add(event.point);
};

eraser.onMouseUp = event => {
    state.path.add(event.point);

    history.add(
        new CanvasAction({
            item: state.path,
        })
    );

    state.path = null;
};

eraser.onActivate = event => {
    ui.activate();
    cursor.activate({
        cursor: new ui.Path.Circle({
            radius: option.eraser.size * 0.5,
            strokeColor: '#282828',
            strokeWidth: 1,
        })
    });
};

eraser.onDeactivate = event => {
    cursor.deactivate();
};

export default eraser;