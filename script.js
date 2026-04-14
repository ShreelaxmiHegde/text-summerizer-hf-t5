document.getElementById("summarization-form").addEventListener("submit", async (e) => {
    e.preventDefault();

    const dialogue = document.getElementById("dialogue-input").value.trim();
    const summary = document.getElementById("summary-text");
    const btn = document.querySelector("button");

    if (!dialogue) return;

    summary.innerText = "Summarizing content...";
    btn.disabled = true;

    try {
        res = await fetch("http://127.0.0.1:8000/summarize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ dialogue })
        });

        if (!res.ok) {
            throw new Error("Error: ", res.status)
        }

        const data = await res.json();
        summary.innerText = data.summary || "No summary returned";
    } catch (err) {
        summary.innerText = err.message;
    } finally {
        btn.disabled = false;
    }

});