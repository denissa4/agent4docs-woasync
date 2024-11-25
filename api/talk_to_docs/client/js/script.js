async function validateToken() {
    const token = document.getElementById("api-token").value;
    if (!token) {
        alert("Please enter an API token.");
        return;
    }

    try {
        const response = await fetch("https://api.nlsql.com/v1/data-source/", {
            method: "GET",
            headers: {
                Authorization: "Token " + token,
            },
        });

        if (response.ok && (await response.text()) !== "not authorised") {
            // Redirect if token is valid
            sessionStorage.setItem("apiToken", token);
            window.location.href = "upload.html";
        } else {
            alert("Invalid API token. Access denied.");
        }
    } catch (error) {
        console.error("Error validating token:", error);
        alert("Failed to validate token. Please try again.");
    }
}
