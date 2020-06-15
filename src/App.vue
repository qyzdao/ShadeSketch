<template>
  <div
    id="app"
    v-hotkey.prevent="keymap"
    @drop.prevent="$refs['artboard'].loadRaster"
    @dragover.prevent
  >
    <div id="main" class="window-left-toolbar">
      <div id="toolbar-container">
        <div id="toolbar">
          <icon-button icon-name="reset" :description="$t('toolbar.reset')" @click="resetApp" />
          <div class="toolbar-sep"></div>
          <icon-button
            icon-name="import"
            :description="$t('toolbar.import')"
            @click="$refs['artboard'].$refs['file'].click()"
          />
          <icon-button
            icon-name="sample"
            :description="$t('toolbar.sample')"
            @click="togglePanel('sample')"
            :active="uiState.panel.sample"
          />
          <div class="toolbar-sep"></div>
          <icon-button
            icon-name="undo"
            :description="$t('toolbar.undo')"
            @click="$refs['artboard'].undo()"
            :disabled="!canvasHistory.canUndo()"
          />
          <icon-button
            icon-name="redo"
            :description="$t('toolbar.redo')"
            @click="$refs['artboard'].redo()"
            :disabled="!canvasHistory.canRedo()"
          />
          <div class="toolbar-sep"></div>
          <icon-button
            icon-name="pen"
            :description="$t('toolbar.brush')"
            @click="toggleTool('pen')"
            :active="uiState.tool.pen"
          />
          <icon-button
            icon-name="eraser"
            :description="$t('toolbar.eraser')"
            @click="toggleTool('eraser')"
            :active="uiState.tool.eraser"
          />
          <div class="toolbar-sep"></div>
          <icon-button
            icon-name="pan"
            :description="$t('toolbar.move')"
            @click="toggleTool('pan')"
            :active="uiState.tool.pan"
          />
          <icon-button
            icon-name="rotate"
            :description="$t('toolbar.rotate')"
            @click="toggleTool('rotate')"
            :active="uiState.tool.rotate"
          />
          <icon-button
            icon-name="zoom"
            :description="$t('toolbar.scale')"
            @click="toggleTool('zoom')"
            :active="uiState.tool.zoom"
          />
          <div class="toolbar-sep"></div>
          <icon-button
            icon-name="preprocess"
            :description="$t('toolbar.preprocess')"
            @click="togglePanel('preprocess')"
            :active="uiState.panel.preprocess"
          />
          <icon-button
            icon-name="shade"
            :description="$t('toolbar.shade')"
            @click="togglePanel('shade')"
            :active="uiState.panel.shade"
          />
          <div class="toolbar-sep"></div>
          <icon-button
            icon-name="help"
            :description="$t('toolbar.help')"
            @click="togglePanel('help')"
            :active="uiState.panel.help"
          />
        </div>
      </div>
      <div id="canvas-container">
        <div id="artboard-container">
          <guide />
          <artboard ref="artboard" :width="imgWidth" :height="imgHeight" />
        </div>
        <canvas id="output-canvas" ref="output" :width="imgWidth" :height="imgHeight" />
      </div>
      <div id="message-container">
        <ul>
          <transition-group name="slide-fade">
            <li v-for="message in uiState.message.slice(-6)" :key="message.id">{{ message.text }}</li>
          </transition-group>
        </ul>
      </div>
      <div id="panel-container">
        <div class="panel" id="panel-help" v-show="uiState.panel.help">
          <div class="panel-title">{{$t("panel.help")}}</div>
          <div class="panel-row">
            <button @click="$i18n.locale='en'">English</button>
            <button @click="$i18n.locale='zh-Hans'">简中</button>
            <button @click="$i18n.locale='zh-Hant'">繁中</button>
            <button @click="$i18n.locale='ja'">日本語</button>
          </div>
          <div style="margin-bottom: 6px;">
            <p>{{$t("help.text[0]")}}</p>
            <br />
            <p>{{$t("help.text[1]")}}</p>
            <p>{{$t("help.text[2]")}}</p>
            <p>{{$t("help.text[3]")}}</p>
            <p>{{$t("help.text[4]")}}</p>
            <p>{{$t("help.text[5]")}}</p>
            <p>{{$t("help.text[6]")}}</p>
            <p>{{$t("help.text[7]")}}</p>
            <br />
            <p>© 2020 ShadeSketch, UMBC, Project HAT.</p>
          </div>
          <div class="panel-row">
            <button @click="openLink($t('help.url[0]'))">{{$t("help.link[0]")}}</button>
            <button @click="openLink($t('help.url[1]'))">{{$t("help.link[1]")}}</button>
            <button @click="openLink($t('help.url[2]'))">{{$t("help.link[2]")}}</button>
          </div>
          <div class="panel-row">
            <button @click="startApp" :disabled="appStarted">{{$t("panel.start")}}</button>
          </div>
        </div>
        <div class="panel" id="panel-sample" v-show="uiState.panel.sample">
          <div class="panel-title">{{$t("panel.sample")}}</div>
          <div class="panel-row">
            <div id="sample-list">
              <div
                class="sample-item"
                v-for="item in sampleFiles"
                :key="item"
                @click="$refs['artboard'].addRaster(item)"
              >
                <img :src="item" draggable="false" />
              </div>
            </div>
          </div>
        </div>
        <div class="panel" id="panel-eraser" v-show="uiState.panel.eraser">
          <div class="panel-title">{{$t("panel.eraser")}}</div>
          <div class="panel-row">
            <div class="panel-col label">
              <span>{{$t("panel.size")}}</span>
            </div>
            <div class="panel-col input">
              <a-input-number
                :min="canvasOption.eraser.sizeMin"
                :max="canvasOption.eraser.sizeMax"
                v-model="canvasOption.eraser.size"
              />
            </div>
            <div class="panel-col slider">
              <a-slider
                :min="canvasOption.eraser.sizeMin"
                :max="canvasOption.eraser.sizeMax"
                :tooltipVisible="false"
                v-model="canvasOption.eraser.size"
              />
            </div>
          </div>
        </div>
        <div class="panel" id="panel-pen" v-show="uiState.panel.pen">
          <div class="panel-title">{{$t("panel.brush")}}</div>
          <div class="panel-row">
            <div class="panel-col label">
              <span>{{$t("panel.size")}}</span>
            </div>
            <div class="panel-col input">
              <a-input-number
                :min="canvasOption.pen.sizeMin"
                :max="canvasOption.pen.sizeMax"
                v-model="canvasOption.pen.size"
              />
            </div>
            <div class="panel-col slider">
              <a-slider
                :min="canvasOption.pen.sizeMin"
                :max="canvasOption.pen.sizeMax"
                :tooltipVisible="false"
                v-model="canvasOption.pen.size"
              />
            </div>
          </div>
          <div class="panel-row">
            <div class="panel-col label">
              <span>{{$t("panel.autoCorrection")}}</span>
            </div>
            <div class="panel-col input">
              <a-checkbox v-model="canvasOption.pen.simplify" />
            </div>
          </div>
        </div>
        <div class="panel" id="panel-preprocess" v-show="uiState.panel.preprocess">
          <div class="panel-title">{{$t("panel.preprocessing")}}</div>
          <div class="panel-row">
            <div class="panel-col label">
              <span>{{$t("panel.level")}}</span>
            </div>
            <div class="panel-col input">
              <a-input-number :min="0" :max="255" v-model="appOption.thresholdValue" />
            </div>
            <div class="panel-col slider">
              <a-slider
                :defaultValue="128"
                :min="0"
                :max="255"
                :tooltipVisible="false"
                v-model="appOption.thresholdValue"
              />
            </div>
          </div>
          <div class="panel-row">
            <button
              @click="runFilter('threshold')"
              :disabled="disableInteraction"
            >{{$t("panel.thresholdLine")}}</button>
          </div>
          <div class="panel-row">
            <button
              @click="runFilter('smooth')"
              :disabled="disableInteraction"
            >{{$t("panel.smoothLine")}}</button>
          </div>
        </div>
        <div class="panel" id="panel-shade" v-show="uiState.panel.shade">
          <div class="panel-title">{{$t("panel.shading")}}</div>
          <div class="panel-row">
            <light-stage ref="light-stage" :width="294" :height="294" />
          </div>
          <div class="panel-row">
            <div class="panel-col label">
              <span>{{$t("panel.normalization")}}</span>
            </div>
            <div class="panel-col input">
              <a-checkbox v-model="appOption.useNormalization" />
            </div>
          </div>
          <div class="panel-row">
            <div class="panel-col label">
              <span>{{$t("panel.shadeResult")}}</span>
            </div>
            <div class="panel-col input">
              <a-checkbox v-model="appOption.showShadeResult" />
            </div>
          </div>
          <div class="panel-row">
            <button
              @click="runFilter('shade')"
              :disabled="disableInteraction"
            >{{$t("panel.shadeLine")}}</button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import * as tf from "@tensorflow/tfjs";

tf.enableProdMode();

import Vue from "vue";
import Slider from "ant-design-vue/lib/slider";
import "ant-design-vue/lib/slider/style/css";
import Checkbox from "ant-design-vue/lib/checkbox";
import "ant-design-vue/lib/checkbox/style/css";
import InputNumber from "ant-design-vue/lib/input-number";
import "ant-design-vue/lib/input-number/style/css";
import VueHotkey from "v-hotkey";

Vue.use(Slider);
Vue.use(Checkbox);
Vue.use(InputNumber);
Vue.use(VueHotkey);

import Guide from "./components/Guide";
import Artboard from "./components/Artboard";
import IconButton from "./components/IconButton";
import LightStage from "./components/LightStage";
import Language from "./components/Language.vue";

import history from "./components/Artboard/history";
import { option } from "./components/Artboard/utils";

const SHOW_MESSAGE = true;

const MAX_WIDTH = 320;
const MAX_HEIGHT = 320;

const LINENORMALIZER_PATH = "assets/models/linenormalizer/model.json";
const LINESHADER_PATH = "assets/models/lineshader/model.json";
const LINESMOOTHER_PATH = "assets/models/linesmoother/model.json";
const SAMPLE_FILES = [
  "assets/images/1.png",
  "assets/images/2.png",
  "assets/images/3.png",
  "assets/images/4.png",
  "assets/images/5.png",
  "assets/images/6.png",
  "assets/images/7.png",
  "assets/images/8.png",
  "assets/images/9.png",
  "assets/images/10.png",
  "assets/images/11.png",
  "assets/images/12.png",
  "assets/images/13.png",
  "assets/images/14.png",
  "assets/images/15.png",
  "assets/images/16.png"
];

const sleep = time => {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      resolve();
    }, time);
  });
};

export default {
  name: "App",
  i18n: Language,
  components: {
    LightStage,
    Artboard,
    IconButton,
    Guide
  },
  mounted() {
    this.cache = {};
    this.sampleFiles = SAMPLE_FILES;
  },
  data() {
    return {
      sampleFiles: [],
      imgWidth: MAX_WIDTH,
      imgHeight: MAX_HEIGHT,
      uiState: {
        panel: {
          sample: false,

          pen: false,
          eraser: false,

          shade: true,
          preprocess: true,

          help: true
        },
        tool: {
          pen: false,
          eraser: false,

          pan: false,
          rotate: false,
          zoom: false
        },
        message: []
      },
      canvasHistory: history,
      canvasOption: option,
      appOption: {
        thresholdValue: 128,
        useNormalization: true,
        showShadeResult: false
      },
      appStarted: false,
      appBusying: false
    };
  },
  computed: {
    keymap() {
      return {
        "ctrl+R": this.resetApp,
        "meta+R": this.resetApp,
        "ctrl+O": () => {
          this.$refs["artboard"].$refs["file"].click();
        },
        "meta+O": () => {
          this.$refs["artboard"].$refs["file"].click();
        },
        "ctrl+Y": () => {
          this.$refs["artboard"].redo();
        },
        "meta+Y": () => {
          this.$refs["artboard"].redo();
        },
        "ctrl+Z": () => {
          this.$refs["artboard"].undo();
        },
        "meta+Z": () => {
          this.$refs["artboard"].undo();
        },
        B: () => {
          this.toggleTool("pen");
        },
        E: () => {
          this.toggleTool("eraser");
        },
        M: () => {
          this.toggleTool("pan");
        },
        R: () => {
          this.toggleTool("rotate");
        },
        S: () => {
          this.toggleTool("zoom");
        },
        F1: () => {
          this.togglePanel("help");
        },
        "[": () => {
          this.changeSize(-1);
        },
        "]": () => {
          this.changeSize(+1);
        },
        "alt+1": () => {
          this.runFilter("threshold");
        },
        "alt+2": () => {
          this.runFilter("smooth");
        },
        "alt+3": () => {
          this.runFilter("shade");
        }
      };
    },
    disableInteraction() {
      return this.appStarted === false || this.appBusying === true;
    }
  },
  watch: {
    "appOption.showShadeResult"(newVal, oldVal) {
      if (!newVal && this.cache["compData"])
        this.$refs["output"]
          .getContext("2d")
          .putImageData(this.cache["compData"], 0, 0);
      else if (newVal && this.cache["shadeData"])
        this.$refs["output"]
          .getContext("2d")
          .putImageData(this.cache["shadeData"], 0, 0);
    }
  },
  methods: {
    async startApp() {
      this.togglePanel("help");

      this.showMessage(`${this.$t("message.loadingProgress")} 0.0%...`);

      let totalProgress = 0;
      const onProgress = progress => {
        totalProgress += progress;
        this.showMessage(
          `${this.$t("message.loadingProgress")} ${(
            (totalProgress / 6.5) *
            100
          ).toFixed(1)}%...`
        );
      };

      try {
        [
          this.modelSmooth,
          this.modelNorm,
          this.modelShade
        ] = await Promise.all([
          tf.loadGraphModel(LINESMOOTHER_PATH, { onProgress }),
          tf.loadGraphModel(LINENORMALIZER_PATH, { onProgress }),
          tf.loadGraphModel(LINESHADER_PATH, { onProgress })
        ]);

        this.showMessage(this.$t("message.loadingFinish"));
        this.appStarted = true;
      } catch (err) {
        this.showMessage(this.$t("message.loadingError"));
        console.log(err);
      }
    },
    resetApp() {
      if (confirm(this.$t("reset"))) {
        this.$refs["artboard"].init();
        this.$refs["output"]
          .getContext("2d")
          .clearRect(0, 0, this.imgWidth, this.imgHeight);
        this.toggleTool();
      }
    },
    showMessage(text) {
      if (SHOW_MESSAGE)
        this.uiState.message.push({
          text,
          id: new Date().getTime() + Math.random()
        });
    },
    changeSize(val) {
      const activeTool = this.$refs["artboard"].tool;

      if (activeTool == "pen" || activeTool == "eraser") {
        this.canvasOption[activeTool].size = Math.max(
          Math.min(
            this.canvasOption[activeTool].size + val,
            this.canvasOption[activeTool].sizeMax
          ),
          this.canvasOption[activeTool].sizeMin
        );
      }
    },
    togglePanel(target) {
      this.uiState.panel[target] = !this.uiState.panel[target];
    },
    openLink(url) {
      window.open(url, "_blank");
    },
    toggleTool(target) {
      const _target = this.$refs["artboard"].tool == target ? null : target;

      this.$refs["artboard"].setTool(_target);

      for (let tool in this.uiState.tool) {
        const active = _target == tool;

        this.uiState.tool[tool] = active;
        if (tool in this.uiState.panel) this.uiState.panel[tool] = active;
      }
    },
    async runFilter(filter) {
      const filterMessage = `message.${filter}`;
      const filterFunction = {
        shade: this.runShade,
        smooth: this.runSmooth,
        threshold: this.runThreshold
      };

      try {
        this.showMessage(
          `${this.$t(filterMessage)}${this.$t("message.executeProgress")}`
        );
        await sleep(500);
        const startTime = new Date().getTime();

        filterFunction[filter]();

        const endTime = new Date().getTime();
        this.showMessage(
          `${this.$t(filterMessage)}${this.$t("message.executeFinish")} ~${(
            (endTime - startTime) /
            1000
          ).toFixed(1)} s`
        );
      } catch (err) {
        this.showMessage(
          `${this.$t(filterMessage)}${this.$t("message.executeError")}`
        );
        console.log(err);
      }
    },
    runThreshold() {
      const thresholdValue = this.appOption.thresholdValue;
      const imgHeight = this.imgHeight;
      const imgWidth = this.imgWidth;
      const dataSize = imgWidth * imgHeight * 4;

      const rawData = this.$refs["artboard"].getRaster();

      const processedData = Uint8ClampedArray.from(rawData.data);
      for (let i = 0; i < dataSize; i += 4) {
        const value = rawData.data[i + 0] < thresholdValue ? 0 : 255;
        processedData[i + 0] = value;
        processedData[i + 1] = value;
        processedData[i + 2] = value;
        processedData[i + 3] = 255;
      }

      this.$refs["artboard"].addRaster(
        {
          data: processedData,
          width: imgWidth,
          height: imgHeight
        },
        true
      );
    },
    runSmooth() {
      const imgHeight = this.imgHeight;
      const imgWidth = this.imgWidth;
      const dataSize = imgWidth * imgHeight * 4;

      const rawData = this.$refs["artboard"].getRaster();

      let line = new Float32Array(dataSize / 4);
      for (let i = 0; i < dataSize; i += 4) {
        line[i / 4] = rawData.data[i + 0] / 255;
      }

      const smoothTensor = tf.tidy(() => {
        const lineTensor = tf.tensor4d(line, [1, imgHeight, imgWidth, 1]);
        return this.modelSmooth.predict(lineTensor);
      });

      const smooth = smoothTensor.dataSync();
      const processedData = Uint8ClampedArray.from(rawData.data);
      for (let i = 0; i < dataSize; i += 4) {
        const value = smooth[i / 4] * 255;
        processedData[i + 0] = value;
        processedData[i + 1] = value;
        processedData[i + 2] = value;
        processedData[i + 3] = 255;
      }

      this.$refs["artboard"].addRaster(
        {
          data: processedData,
          width: imgWidth,
          height: imgHeight
        },
        true
      );

      smoothTensor.dispose();
    },
    runShade() {
      const imgHeight = this.imgHeight;
      const imgWidth = this.imgWidth;
      const dataSize = imgWidth * imgHeight * 4;

      const lightPosition = this.$refs["light-stage"].lightPosition;

      const rawData = this.$refs["artboard"].getRaster();

      let line = new Float32Array(dataSize / 4);
      for (let i = 0; i < dataSize; i += 4)
        line[i / 4] = rawData.data[i + 0] / 255.0;

      const shadeInvTensor = tf.tidy(() => {
        const condTensor = tf.tensor2d(lightPosition, [1, 3]);

        const lineTensor = tf.tensor4d(line, [1, imgHeight, imgWidth, 1]);
        const normTensor = this.appOption.useNormalization
          ? this.modelNorm.predict(lineTensor)
          : lineTensor;
        const lineInvTensor = tf.scalar(1.0).sub(normTensor);

        return this.modelShade.predict([condTensor, lineInvTensor]);
      });

      const shadeInv = shadeInvTensor.dataSync();

      const shadeData = new ImageData(imgWidth, imgHeight);
      const compData = new ImageData(imgWidth, imgHeight);
      for (let i = 0; i < dataSize; i += 4) {
        const shade = (1 - (shadeInv[i / 4] + 1) * 0.5) * 255;
        shadeData.data[i + 0] = shade;
        shadeData.data[i + 1] = shade;
        shadeData.data[i + 2] = shade;
        shadeData.data[i + 3] = 255;

        const comp = rawData.data[i + 0] * 0.8 + shade * 0.2;
        compData.data[i + 0] = comp;
        compData.data[i + 1] = comp;
        compData.data[i + 2] = comp;
        compData.data[i + 3] = 255;
      }

      this.cache["shadeData"] = shadeData;
      this.cache["compData"] = compData;

      if (!this.appOption.showShadeResult)
        this.$refs["output"].getContext("2d").putImageData(compData, 0, 0);
      else this.$refs["output"].getContext("2d").putImageData(shadeData, 0, 0);

      shadeInvTensor.dispose();
    }
  }
};
</script>

<style>
body {
  background: none;
  margin: 0;
  color: rgba(0, 0, 0, 0.65);
  font-size: 12px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
    "Hiragino Sans GB", "Microsoft YaHei", "Helvetica Neue", Helvetica, Arial,
    sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
  font-variant: tabular-nums;
  line-height: 1.5;
  font-feature-settings: "tnum";
}
#canvas-container canvas {
  box-shadow: rgb(0, 0, 0, 0.2) 0px 0px 20px;
}
#app {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  justify-content: center;
  align-items: center;
  overflow: hidden;
}
#main {
  margin-left: 44px;
  margin-right: 320px;
}
#main.side-toolbar {
  display: flex;
  flex-direction: row;
}
#toolbar-container {
  display: flex;
  margin: 0 40px;
  z-index: 999999;
}
#toolbar {
  background: rgb(64, 64, 64);
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 0 3px;
}
.side-toolbar #toolbar-container {
  margin: 0 20px 0 -60px;
}
.side-toolbar #toolbar {
  flex-direction: column;
  padding: 3px 0;
}
.window-top-toolbar #toolbar-container {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  display: block;
  margin: 0;
}
.window-top-toolbar #toolbar {
  justify-content: flex-start;
}
.window-left-toolbar #toolbar-container {
  position: fixed;
  top: 0;
  left: 0;
  bottom: 0;
  display: block;
  margin: 0;
}
.window-left-toolbar #toolbar {
  flex-direction: column;
  height: 100%;
  justify-content: flex-start;
  padding: 3px 0;
  min-height: 480px;
}
.toolbar-sep {
  background: rgb(85, 85, 85);
  margin: 0px 1px;
  width: 1px;
  height: 30px;
}
.side-toolbar .toolbar-sep,
.window-left-toolbar .toolbar-sep {
  height: 1px;
  width: 30px;
}
.side-toolbar .icon-button,
.window-left-toolbar .icon-button {
  margin: 3px 6px;
}
#canvas-container {
  display: flex;
  justify-content: center;
  align-items: center;
}
#artboard-container {
  position: relative;
  margin: 40px;
}
#output-canvas {
  background: rgb(255, 255, 255);
  margin: 40px;
}
#panel-container {
  position: fixed;
  top: 0;
  right: 0;
  bottom: 0;
  display: flex;
  flex-direction: column;
  width: 320px;
  background: rgb(64, 64, 64);
  overflow-y: auto;
  overflow-x: hidden;
  user-select: none;
}
#panel-container {
  -ms-overflow-style: none;
  scrollbar-width: none;
}
#panel-container::-webkit-scrollbar {
  display: none;
}
.panel {
  margin: 7px;
  background: rgb(73, 73, 73);
  padding: 6px 6px 0px 6px;
  border-radius: 2px;
  color: rgb(187, 187, 187);
}
.panel-row {
  margin-bottom: 6px;
  display: flex;
  align-items: center;
}
.panel-col {
  display: block;
}
.panel-col.label {
  width: 33.33%;
}
.panel-col.input {
  width: 16.67%;
}
.panel-col.slider {
  width: 50%;
}
.panel-title {
  font-weight: bold;
  padding-bottom: 2px;
  margin-bottom: 4px;
  background: rgb(93, 93, 93);
  padding-left: 4px;
}
button {
  width: 100%;
  color: rgb(200, 200, 200);
  background-color: rgb(93, 93, 93);
  cursor: pointer;
  display: inline-block;
  text-align: center;
  vertical-align: middle;
  outline: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
  border: 1px solid rgb(98, 98, 98);
  padding: 6px 12px;
  font-size: 12px;
  line-height: 1.5;
  transition: color 0.01s ease-in-out, background-color 0.01s ease-in-out,
    border-color 0.01s ease-in-out, box-shadow 0.01s ease-in-out;
}
button:hover {
  background-color: rgb(112, 112, 112);
  border: 1px solid rgb(117, 117, 117);
}
button:active {
  background-color: rgb(29, 29, 29);
  border: 1px solid rgb(34, 34, 34);
}
button:disabled {
  color: rgb(110, 110, 110);
  cursor: default;
}
button:disabled:hover {
  background-color: rgb(93, 93, 93);
  border: 1px solid rgb(98, 98, 98);
}
.ant-checkbox-inner {
  background: rgb(41, 41, 41);
  border-color: rgb(41, 41, 41);
  width: 14px;
  height: 14px;
}

.ant-checkbox-checked .ant-checkbox-inner {
  background-color: rgb(41, 41, 41);
  border-color: rgb(41, 41, 41);
}

.ant-checkbox:hover::after,
.ant-checkbox-wrapper:hover .ant-checkbox::after {
  visibility: hidden;
}
.ant-checkbox-wrapper:hover .ant-checkbox-inner,
.ant-checkbox:hover .ant-checkbox-inner,
.ant-checkbox-input:focus + .ant-checkbox-inner {
  border-color: rgb(41, 41, 41);
}

.ant-input-number-handler-wrap {
  visibility: hidden;
  width: 0;
}
.ant-input-number {
  width: 100%;
  background-color: rgb(41, 41, 41);
  border-radius: 3px;
  border: none;
  color: rgb(187, 187, 187);
  font-size: 12px;
  height: 22px;
}
.ant-input-number-input {
  height: 24px;
  padding: 0px 4px;
}
.ant-input-number:focus,
.ant-input-number-focused {
  border-color: rgb(41, 41, 41);
  border-right-width: 1px !important;
  outline: 0;
  box-shadow: none;
}
.ant-slider-track,
.ant-slider:hover .ant-slider-track {
  background-color: rgba(0, 0, 0, 0);
}
.ant-slider-rail,
.ant-slider:hover .ant-slider-rail {
  background-color: rgb(41, 41, 41);
}
.ant-slider-handle {
  border: none;
  border-radius: 2px;
  background-color: rgb(189, 189, 189);
  width: 7px;
}
.ant-slider-handle:focus {
  box-shadow: none;
}
.ant-slider {
  margin: 6px;
}
#sample-list {
  display: flex;
  flex-wrap: wrap;
  margin-left: -2px;
  margin-top: -2px;
}
.sample-item {
  border-radius: 2px;
  overflow: hidden;
  margin-left: 2px;
  margin-top: 2px;
  width: 72px;
  background-color: rgb(98, 98, 98);
}
.sample-item img {
  width: 100%;
}
p {
  margin-bottom: 6px;
}
#message-container {
  position: fixed;
  left: 44px;
  top: 0;
  bottom: 24px;
  display: flex;
  flex-direction: column-reverse;
  width: 320px;
  user-select: none;
  pointer-events: none;
  color: rgb(156, 156, 156);
}
#message-container ul {
  list-style: disc;
  margin: 0;
}
#message-container li {
  margin-top: 6px;
}
.slide-fade-enter-active {
  transition: all 0.5s ease;
}
.slide-fade-leave-active {
  transition: all 0.5s ease;
}
.slide-fade-enter,
.slide-fade-leave-to {
  transform: translateX(30px);
  opacity: 0;
}
</style>