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
                                        "segments": [[[10005, 10000.83333], [0, 0], [0, 2.75917]], [[10000, 10005.83333], [2.75917, 0], [-2.75833, 0]], [[9995, 10000.83333], [0, 2.75917], [0, -2.75833]], [[10000, 9995.83333], [-2.75833, 0], [0, 0]], [10000, 9999.16667], [10004.16667, 9995], [10000, 9990.83333], [[10000, 9994.16667], [0, 0], [-3.68333, 0]], [[9993.33333, 10000.83333], [0, -3.68333], [0, 3.68333]], [[10000, 10007.5], [-3.68333, 0], [3.68333, 0]], [[10006.66667, 10000.83333], [0, 3.68333], [0, 0]]],
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

const rotate = new paper.Tool();

rotate.onMouseDown = event => {
    state.prev = paper.project.view.rotation;
    state.start = event.point;
};

rotate.onMouseDrag = event => {
    if (!state.start) return;

    const center = paper.project.view.center;

    const fixAxis = Math.atan2(state.start.y - center.y, state.start.x - center.x) * 180 / Math.PI;
    const rotAxis = Math.atan2(event.point.y - center.y, event.point.x - center.x) * 180 / Math.PI;

    const angle = rotAxis - fixAxis;

    paper.project.view.rotate(angle);
};

rotate.onMouseUp = event => {
    history.add(
        new ViewAction({
            rotation: {
                prev: state.prev,
                next: paper.project.view.rotation,
            }
        })
    );
};

rotate.onActivate = () => {
    ui.activate();
    cursor.activate({
        cursor: (new ui.Group()).importJSON(CURSOR_JSON)
    });
};

rotate.onDeactivate = event => {
    cursor.deactivate();
};

export default rotate;