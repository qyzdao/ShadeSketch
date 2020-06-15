import {
    canvas as paper,
    cursor,
    ui,
} from "../utils";
import history, {
    ViewAction
} from "../history";

const CURSOR_JSON = `["Group", {
    "applyMatrix": true,
    "children": [["Group", {
                "applyMatrix": true,
                "children": [["Shape", {
                            "applyMatrix": false,
                            "matrix": [0.83333, 0, 0, 0.83333, 10000, 10000],
                            "clipMask": true,
                            "type": "rectangle",
                            "size": [24, 24],
                            "radius": [0, 0]
                        }
                    ], ["Group", {
                            "applyMatrix": true,
                            "children": [["Path", {
                                        "applyMatrix": true,
                                        "segments": [[10000.83333, 9995], [10000.83333, 9999.16667], [10005, 9999.16667], [10005, 9996.45833], [10008.54167, 10000], [10005, 10003.54167], [10005, 10000.83333], [10000.83333, 10000.83333], [10000.83333, 10005], [10003.54167, 10005], [10000, 10008.54167], [9996.45833, 10005], [9999.16667, 10005], [9999.16667, 10000.83333], [9995, 10000.83333], [9995, 10003.54167], [9991.45833, 10000], [9995, 9996.45833], [9995, 9999.16667], [9999.16667, 9999.16667], [9999.16667, 9995], [9996.45833, 9995], [10000, 9991.45833], [10003.54167, 9995]],
                                        "closed": true,
                                        "fillColor": [0.15, 0.15, 0.15, 1]
                                    }
                                ]]
                        }
                    ]]
            }
        ]]
}]`;

const state = {
    start: null,
    prev: null,
};

const pan = new paper.Tool();

pan.onMouseDown = event => {
    state.start = event.point;
    state.prev = paper.project.view.center;
};

pan.onMouseDrag = event => {
    if (!state.start) return;

    paper.project.view.translate(event.point.subtract(state.start));
};

pan.onMouseUp = event => {
    history.add(
        new ViewAction({
            center: {
                prev: state.prev,
                next: paper.project.view.center,
            }
        })
    );
};

pan.onActivate = () => {
    ui.activate()
    cursor.activate({
        cursor: (new ui.Group()).importJSON(CURSOR_JSON)
    });
};

pan.onDeactivate = event => {
    cursor.deactivate();
};

export default pan;