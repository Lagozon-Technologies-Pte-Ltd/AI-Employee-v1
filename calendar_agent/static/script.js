document.getElementById("toggle-raw").addEventListener("click", function() {
    const rawDataDiv = document.getElementById("raw-data");
    if (rawDataDiv.style.display === "none") {
        rawDataDiv.style.display = "block";
        this.textContent = "Hide Raw Data";
    } else {
        rawDataDiv.style.display = "none";
        this.textContent = "Show Raw Data";
    }
});