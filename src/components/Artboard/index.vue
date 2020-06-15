<template>
  <div :class="{'default-cursor': !tool}">
    <input ref="file" type="file" @change="loadRaster" />
    <canvas ref="canvas" id="canvas" @drop.prevent="loadRaster" @dragover.prevent />
    <canvas ref="ui" id="ui" />
  </div>
</template>

<script>
import { canvas as paper, option, cursor, ui, canvas } from "./utils";
import history, { CanvasAction } from "./history";
import tools from "./tools";

const MAX_WIDTH = 4096;
const MAX_HEIGHT = 4096;

export default {
  name: "Artboard",
  props: {
    height: {
      default: 320,
      type: Number
    },
    width: {
      default: 320,
      type: Number
    }
  },
  components: {
    event
  },
  mounted() {
    //override hdpi device pixel ratio to ignore paperjs scale
    window.devicePixelRatio = 1;

    ui.setup(this.$refs["ui"]);
    paper.setup(this.$refs["canvas"]);

    this.init();
  },
  data() {
    return {
      tool: null,
      option,
      history
    };
  },
  watch: {
    "option.pen.size"(newVal, oldVal) {
      this.setCursor(newVal);
    },
    "option.eraser.size"(newVal, oldVal) {
      this.setCursor(newVal);
    },
    height(newVal, oldVal) {
      ui.view.viewSize = new paper.Size(this.width, newVal);
      paper.view.viewSize = new paper.Size(this.width, newVal);
    },
    width(newVal, oldVal) {
      ui.view.viewSize = new paper.Size(newVal, this.height);
      paper.view.viewSize = new paper.Size(newVal, this.height);
    }
  },
  methods: {
    init() {
      ui.project.clear();
      paper.project.clear();

      ui.view.viewSize = new paper.Size(this.width, this.height);
      paper.view.viewSize = new paper.Size(this.width, this.height);

      paper.view.center = new paper.Point(0, 0);
      paper.view.zoom = 1;
      paper.view.rotation = 0;

      new paper.Shape.Rectangle({
        point: [
          -MAX_WIDTH / 2 + this.width / 2,
          -MAX_HEIGHT / 2 + this.height / 2
        ],
        size: [MAX_WIDTH, MAX_HEIGHT],
        strokeWidth: 0,
        fillColor: "#ffffff"
      });

      tools["pointer"].activate();
      this.tool = null;

      history.init();
      cursor.init();
    },
    undo() {
      history.undo();
    },
    redo() {
      history.redo();
    },
    hideCursor() {
      cursor.hide();
    },
    showCursor() {
      cursor.show();
    },
    setCursor(arg) {
      ui.activate();
      cursor.activate({
        cursor: new ui.Path.Circle({
          radius: arg * 0.5,
          strokeColor: "#282828",
          strokeWidth: 1
        })
      });
    },
    setTool(toolName) {
      if (!!toolName) {
        tools[toolName].activate();
        this.tool = toolName;
      } else {
        tools["pointer"].activate();
        this.tool = null;
      }
    },
    addRaster(src, internal = false) {
      let raster = null;

      if (typeof src === "string") {
        raster = new paper.Raster({
          source: src,
          position: paper.view.center
        });
      } else {
        const { data, width, height } = src;

        raster = new paper.Raster({
          size: new paper.Size(width, height),
          position: paper.view.center
        });

        raster.setImageData(new ImageData(data, width, height));
      }

      if (raster && internal) {
        raster.scale(1 / paper.view.zoom);
        raster.rotate(-paper.view.rotation);
      }

      history.add(
        new CanvasAction({
          item: raster
        })
      );
    },
    loadRaster(ev) {
      const reader = new FileReader();
      reader.onload = event => this.addRaster(event.target.result);

      if (ev.dataTransfer && ev.dataTransfer.files[0]) {
        reader.readAsDataURL(ev.dataTransfer.files[0]);
        ev.target.value = null;
      } else if (ev.target && ev.target.files[0]) {
        reader.readAsDataURL(ev.target.files[0]);
        ev.target.value = null;
      }
    },
    getRaster() {
      const rawData = this.$refs["canvas"]
        .getContext("2d")
        .getImageData(0, 0, this.width, this.height);

      return rawData;
    }
  }
};
</script>

<style scoped>
input[type="file"] {
  display: none;
}
#canvas {
  --square-color: rgb(204, 204, 204);
  background-color: rgb(255, 255, 255);
  background-image: linear-gradient(
      45deg,
      var(--square-color) 26%,
      rgba(0, 0, 0, 0) 0,
      rgba(0, 0, 0, 0) 75%,
      var(--square-color) 0,
      var(--square-color)
    ),
    linear-gradient(
      45deg,
      var(--square-color) 26%,
      rgba(0, 0, 0, 0) 0,
      rgba(0, 0, 0, 0) 75%,
      var(--square-color) 0,
      var(--square-color)
    );
  background-position: -2px -2px, 4px 4px;
  background-size: 12px 12px;
  cursor: none;
}
#ui {
  position: absolute;
  left: 0;
  top: 0;
  background-image: none;
  background-color: rgba(0, 0, 0, 0);
  box-shadow: none;
  pointer-events: none;
}
.default-cursor #canvas {
  cursor: default;
}
</style>