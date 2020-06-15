<template>
  <div id="artboard-guide">
    <div v-if="!history.canUndo()">
      <svg
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
      >
        <path
          d="M21 15V19C21 19.5304 20.7893 20.0391 20.4142 20.4142C20.0391 20.7893 19.5304 21 19 21H5C4.46957 21 3.96086 20.7893 3.58579 20.4142C3.21071 20.0391 3 19.5304 3 19V15"
          stroke-width="2"
          stroke-linecap="round"
          stroke-linejoin="round"
        />
        <path d="M17 8L12 3L7 8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
        <path d="M12 3V15" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" />
      </svg>
      <div>
        {{$t('guide.start[0]')}}
        <br />
        {{$t('guide.start[1]')}}
      </div>
    </div>
  </div>
</template>

<script>
import history from "./Artboard/history";

export default {
  name: "Guide",
  data() {
    return {
      history
    };
  }
};
</script>

<style scoped>
#artboard-guide {
  z-index: 99999;
  position: absolute;
  top: 40px;
  left: 40px;
  right: 40px;
  bottom: 40px;
  border: 1px solid rgba(0, 0, 0, 0.1);
  pointer-events: none;
  user-select: none;
  display: flex;
  justify-content: center;
  align-items: center;
}
#artboard-guide div {
  text-align: center;
}
path {
  stroke: rgba(0, 0, 0, 0.3);
}
</style>>