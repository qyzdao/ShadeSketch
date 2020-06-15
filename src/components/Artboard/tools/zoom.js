import {
    canvas as paper,
    cursor,
    ui,
} from "../utils";
import history, {
    ViewAction
} from "../history";

const DELTA_ZOOM = 0.025;
const MIN_OFFSET_Y = 5;

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
                            "children": [["CompoundPath", {
                                        "applyMatrix": true,
                                        "children": [["Path", {
                                                    "applyMatrix": true,
                                                    "segments": [[9998.825, 10000], [10002.15833, 9996.66667], [9999.16667, 9996.66667], [9999.16667, 9995], [10005, 9995], [10005, 10000.83333], [10003.33333, 10000.83333], [10003.33333, 9997.84167], [10000, 10001.175], [10000, 10003.33333], [10006.66667, 10003.33333], [10006.66667, 9993.33333], [9996.66667, 9993.33333], [9996.66667, 10000], [9998.825, 10000]],
                                                    "fillColor": [0.15, 0.15, 0.15, 1]
                                                }
                                            ], ["Path", {
                                                    "applyMatrix": true,
                                                    "segments": [[10008.33333, 9991.66667], [10008.33333, 10005], [10000, 10005], [10000, 10008.33333], [9991.66667, 10008.33333], [9991.66667, 10000], [9995, 10000], [9995, 9991.66667], [10008.33333, 9991.66667]],
                                                    "fillColor": [0.15, 0.15, 0.15, 1]
                                                }
                                            ], ["Path", {
                                                    "applyMatrix": true,
                                                    "segments": [[9998.33333, 10001.66667], [9993.33333, 10001.66667], [9993.33333, 10006.66667], [9998.33333, 10006.66667]],
                                                    "closed": true,
                                                    "fillColor": [0.15, 0.15, 0.15, 1]
                                                }
                                            ]],
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

const zoom = new paper.Tool();

zoom.onMouseDown = event => {
    state.prev = paper.project.view.zoom;
    state.start = paper.project.view.projectToView(event.point);
};

zoom.onMouseDrag = event => {
    if (!state.start) return;

    const point = paper.project.view.projectToView(event.point);
    const direction = point.subtract(state.start).y > 0;

    if (Math.abs(point.subtract(state.start).y) > MIN_OFFSET_Y) {
        paper.project.view.scale(direction ? (1 - DELTA_ZOOM) : (1 + DELTA_ZOOM));

        state.start = point;
    }
};

zoom.onMouseUp = event => {
    history.add(
        new ViewAction({
            zoom: {
                prev: state.prev,
                next: paper.project.view.zoom,
            }
        })
    );
};

zoom.onActivate = () => {
    ui.activate();
    cursor.activate({
        cursor: (new ui.Group()).importJSON(CURSOR_JSON)
    });
};

zoom.onDeactivate = event => {
    cursor.deactivate();
};

export default zoom;