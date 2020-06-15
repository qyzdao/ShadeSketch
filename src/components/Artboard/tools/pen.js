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

const pen = new paper.Tool();

pen.onMouseDown = event => {
    state.path = new paper.Path();
    state.path.strokeColor = option.pen.color;
    state.path.strokeWidth = option.pen.size / paper.project.view.zoom;
    state.path.strokeCap = "round";
    state.path.strokeJoin = "round";

    state.path.add(event.point);
};

pen.onMouseDrag = event => {
    if (!state.path) return;

    state.path.add(event.point);
};

pen.onMouseUp = event => {
    state.path.add(event.point);

    if (option.pen.simplify)
        state.path.simplify(10.0);

    history.add(
        new CanvasAction({
            item: state.path,
        })
    );

    state.path = null;
};

pen.onActivate = event => {
    ui.activate();
    cursor.activate({
        cursor: new ui.Path.Circle({
            radius: option.pen.size * 0.5,
            strokeColor: '#282828',
            strokeWidth: 1,
        })
    });
};

pen.onDeactivate = event => {
    cursor.deactivate();
};

export default pen;