<script>
import Vue from "vue";
import VueI18n from "vue-i18n";

import en from "../locales/en.json";
import ja from "../locales/ja.json";
import zhHans from "../locales/zh-Hans.json";
import zhHant from "../locales/zh-Hant.json";

Vue.use(VueI18n);

export default new VueI18n({
  locale: navigator.language || "en",
  fallbackLocale: "en",
  messages: {
    en: { ...en },
    ja: { ...ja },
    "zh-Hans": { ...zhHans },
    "zh-CN": { ...zhHans },
    "zh-Hant": { ...zhHant },
    "zh-TW": { ...zhHant }
  }
});
</script>