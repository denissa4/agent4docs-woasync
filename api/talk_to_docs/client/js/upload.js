const token = sessionStorage.getItem("apiToken");

async function validate() {
    if (!token) {
        window.location.href = "index.html";
    } else {
        try {
            const response = await fetch("https://api.nlsql.com/v1/data-source/", {
                method: "GET",
                headers: {
                    Authorization: "Token " + token,
                },
            });

            if (response.ok && (await response.text()) !== "not authorised") {
                sessionStorage.setItem("apiToken", token);
            } else {
                alert("Access denied. Please enter a valid API token");
                window.location.href = "index.html";
            }
        } catch (error) {
            console.error("Error validating token:", error);
            alert("Failed to validate token. Please try again.");
            window.location.href = "index.html";
        }
    }
}

async function uploadFiles(event) {
    event.preventDefault();
    const el = document.getElementById("file-upload");
    const files = el.files;

    const hostname = window.location.hostname;
    const protocol = window.location.protocol;

    const allowedTypes = [
        "application/pdf", // PDF
        "text/plain", // Plain text
        "application/msword", // Microsoft Word (.doc)
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document", // Microsoft Word (.docx)
        "text/csv", // CSV
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", // Excel (.xlsx)]
    ];

    const formData = new FormData();
    const feedback = document.getElementById("feedback");

    feedback.innerHTML = "";

    for (const file of files) {
        if (allowedTypes.includes(file.type)) {
            formData.append("files[]", file);
        } else {
            alert(`File type not allowed: ${file.type}. Please upload a PDF, TXT, or Word document.`);
        }
    }

    const response = await fetch(`${protocol}//${hostname}/upload`, {
        method: "POST",
        body: formData,
        // mode: "no-cors",
    })
        .then((response) => {
            if (!response.ok) {
                throw new Error("Network response was not ok " + response.statusText);
            }
            return response.json(); // Parse the response as JSON
        })
        .then((data) => {
            console.log("Success:", data); // Log the response data
            feedback.innerHTML = "Your documents have been uploaded successfully.";
            clearFeedback();
        })
        .catch((error) => {
            console.error("Error:", error); // Log any errors
            feedback.innerHTML = "There was an error uploading your documents.";
            clearFeedback();
        });
}

function clearFeedback() {
    setTimeout(() => {
        feedback.innerHTML = "";
    }, 5000);
}

// Call validate() on page load to verify user's API token
validate();
