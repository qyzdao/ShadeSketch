<template>
  <canvas :width="width" :height="height" style="outline:none;"></canvas>
</template>

<script>
import { Engine } from "@babylonjs/core/Engines/engine";
import { Scene } from "@babylonjs/core/scene";
import { Vector3, Color3 } from "@babylonjs/core/Maths/math";
import { ArcRotateCamera } from "@babylonjs/core/Cameras/arcRotateCamera";
import { HemisphericLight } from "@babylonjs/core/Lights/hemisphericLight";
import { StandardMaterial } from "@babylonjs/core/Materials/standardMaterial";
import { BoundingBoxRenderer } from "@babylonjs/core/Rendering/boundingBoxRenderer";
import { PointerDragBehavior } from "@babylonjs/core/Behaviors/Meshes/pointerDragBehavior";
import { GizmoManager } from "@babylonjs/core/Gizmos";
import { Mesh } from "@babylonjs/core/Meshes/mesh";
import { TextBlock } from "@babylonjs/gui/2D/controls/textBlock";
import { AdvancedDynamicTexture } from "@babylonjs/gui/2D/advancedDynamicTexture";
import "@babylonjs/core/Meshes/meshBuilder";
import { MeshBuilder } from "@babylonjs/core/Meshes/meshBuilder";

const USE_ARBITRARY_POSITION = false;
const DEFAULT_LIGHT_POSITION = [-1, 1, -1];

const showLightPosition = (textObj, lightObj) => {
  const { x, y, z } = lightObj.position;

  textObj.text = `{${(x / 1.5).toFixed(1)},${(y / 1.5).toFixed(1)},${(
    z / 1.5
  ).toFixed(1)}}`;
};

const makePointerDragBehavior = (position, callback) => {
  const pointerDragBehavior = new PointerDragBehavior({
    dragAxis: new Vector3(0, 0, 0)
  });
  pointerDragBehavior.useObjectOrientationForDragging = false;
  pointerDragBehavior.onDragStartObservable.add(event => callback(position));

  return pointerDragBehavior;
};

const createScene = (canvas, engine, callback) => {
  // Create scene
  const scene = new Scene(engine);
  // scene.clearColor = new Color3(1, 1, 1);
  scene.clearColor = new Color3(0.35, 0.35, 0.35);
  // Create camera
  const camera = new ArcRotateCamera(
    "Camera",
    -Math.PI / 3,
    Math.PI / 2.5,
    7.5,
    Vector3.Zero(),
    scene
  );
  camera.attachControl(canvas, true);
  camera.inputs.removeByType("ArcRotateCameraMouseWheelInput");

  // Create light
  const light = new HemisphericLight(
    "HemiLight",
    new Vector3(...DEFAULT_LIGHT_POSITION),
    scene
  );

  // Create objects
  const lightObj = Mesh.CreateSphere("sphereGlass", 48, 0.2, scene);
  lightObj.translate(new Vector3(...DEFAULT_LIGHT_POSITION), 1.5);

  const lightMat = new StandardMaterial("lightMat", scene);
  lightMat.emissiveColor = new Color3(0.98, 0.83, 0.11);
  lightMat.disableLighting = true;
  lightMat.alpha = 1.0;
  lightObj.material = lightMat;

  if (USE_ARBITRARY_POSITION) {
    const gizmoManager = new GizmoManager(scene);
    gizmoManager.positionGizmoEnabled = true;
    gizmoManager.gizmos.positionGizmo.onDragEndObservable.add(event => {
      callback([
        lightObj.position.x / 1.5,
        lightObj.position.y / 1.5,
        lightObj.position.z / 1.5
      ]);

      light.direction = lightObj.position;

      showLightPosition(positionIndicator, lightObj);
    });
    gizmoManager.attachableMeshes = [lightObj];
    gizmoManager.attachToMesh(lightObj);
  }

  const centerObj = MeshBuilder.CreateTorusKnot(
    "center",
    {
      radius: 0.3,
      tube: 0.12,
      tubularSegments: 64,
      radialSegments: 64
    },
    scene
  );
  centerObj.translate(new Vector3(0, 0, 0), 0);

  const centerMat = new StandardMaterial("cel", scene);
  centerMat.diffuseColor = new Color3(0.9, 0.9, 0.9);
  centerObj.material = centerMat;

  const boxFrameObj = MeshBuilder.CreateBox(
    "boxFrame",
    {
      width: 3,
      height: 3,
      depth: 3
    },
    scene
  );

  const boxFrameMat = new StandardMaterial("boxFrameMat", scene);
  boxFrameMat.emissiveColor = new Color3(0.58, 0.46, 0.64);
  boxFrameMat.disableLighting = true;
  boxFrameMat.alpha = 0.4;
  boxFrameObj.material = boxFrameMat;
  boxFrameObj.showBoundingBox = true;
  boxFrameObj.isPickable = false;

  const sphereFrameAPos = [
    [0, 0, -1],
    [0, 1, -1],
    // [1, 1, -1],
    [1, 0, -1],
    // [1, -1, -1],
    [0, -1, -1],
    // [-1, -1, -1],
    [-1, 0, -1],
    // [-1, 1, -1],
    [0, 1, 0],
    [1, 1, 0],
    [1, 0, 0],
    [1, -1, 0],
    [0, -1, 0],
    [-1, -1, 0],
    [-1, 0, 0],
    [-1, 1, 0],
    [0, 1, 1],
    // [1, 1, 1],
    [1, 0, 1],
    // [1, -1, 1],
    [0, -1, 1],
    // [-1, -1, 1],
    [-1, 0, 1],
    // [-1, 1, 1],
    [0, 0, 1]
  ];

  const sphereFrameBPos = [
    // [0, 1, -1],
    [1, 1, -1],
    // [1, 0, -1],
    [1, -1, -1],
    // [0, -1, -1],
    [-1, -1, -1],
    // [-1, 0, -1],
    [-1, 1, -1],
    // [0, 1, 1],
    [1, 1, 1],
    // [1, 0, 1],
    [1, -1, 1],
    // [0, -1, 1],
    [-1, -1, 1],
    // [-1, 0, 1],
    [-1, 1, 1]
  ];

  const sphereFrameMat = new StandardMaterial("sphereFrameMat", scene);
  sphereFrameMat.emissiveColor = new Color3(0.78, 0.7, 0.81);
  sphereFrameMat.disableLighting = true;
  sphereFrameMat.alpha = 1.0;

  const pdbCallback = position => {
    // console.log(position);
    callback(position);

    lightObj.position = new Vector3(
      position[0] * 1.5,
      position[1] * 1.5,
      position[2] * 1.5
    );

    light.direction = lightObj.position;

    showLightPosition(positionIndicator, lightObj);
  };

  const sphereFrameAObjs = sphereFrameAPos.map((pos, idx) => {
    const obj = Mesh.CreateSphere(`sphereFrameObjs_${idx}`, 8, 0.12, scene);
    obj.translate(new Vector3(pos[0], pos[1], pos[2]), 1.5);
    obj.material = sphereFrameMat;

    const pointerDragBehavior = makePointerDragBehavior(pos, pdbCallback);
    obj.addBehavior(pointerDragBehavior);

    return obj;
  });

  const sphereFrameBObjs = sphereFrameBPos.map((pos, idx) => {
    const obj = Mesh.CreateSphere(`sphereFrameObjs_${idx}`, 8, 0.16, scene);
    obj.translate(new Vector3(pos[0], pos[1], pos[2]), 1.5);
    obj.material = sphereFrameMat;

    const pointerDragBehavior = makePointerDragBehavior(pos, pdbCallback);
    obj.addBehavior(pointerDragBehavior);

    return obj;
  });

  // GUI
  const advancedTexture = AdvancedDynamicTexture.CreateFullscreenUI("UI");

  const positionIndicator = new TextBlock();
  //positionIndicator.text = "[0.1,0.5,0.5]";
  positionIndicator.color = "white";
  positionIndicator.alpha = 0.75;
  positionIndicator.fontSize = 12;
  showLightPosition(positionIndicator, lightObj);
  advancedTexture.addControl(positionIndicator);

  positionIndicator.linkWithMesh(lightObj);
  positionIndicator.linkOffsetY = -20;

  const eyeIndicator = new TextBlock();
  eyeIndicator.text = "Eye";
  eyeIndicator.color = "white";
  eyeIndicator.alpha = 0.75;
  eyeIndicator.fontSize = 16;
  advancedTexture.addControl(eyeIndicator);

  eyeIndicator.linkWithMesh(sphereFrameAObjs[0]);
  eyeIndicator.linkOffsetX = -20;
  eyeIndicator.linkOffsetY = -0;

  return scene;
};

export default {
  name: "LightStage",
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
  mounted() {
    this.canvas = this.$el;
    this.engine = new Engine(this.canvas, true);
    this.scene = createScene(this.canvas, this.engine, newPos => {
      this.lightPosition = newPos;
    });

    this.engine.runRenderLoop(() => {
      this.scene.render();
    });

    this.lightPosition = DEFAULT_LIGHT_POSITION;
  }
};
</script>

<style>
</style>