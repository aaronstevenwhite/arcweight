// Fix mathcal rendering issue on macOS by forcing web fonts instead of local STIX fonts
MathJax.Hub.Config({
  "HTML-CSS": {
    availableFonts: ["TeX"],
    preferredFont: "TeX",
    webFont: "TeX"
  }
});