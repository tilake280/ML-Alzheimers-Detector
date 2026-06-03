// --- VIEW TOGGLE LOGIC ---
function toggleMode(mode) {
    const clinicalSection = document.getElementById('clinical-section'); // MAKE SURE your main app has id="clinical-section"
    const gameSection = document.getElementById('game-section');
    
    if (mode === 'game') {
        clinicalSection.style.display = 'none';
        gameSection.style.display = 'block';
        initGame(); // Start game when tab is clicked
    } else {
        clinicalSection.style.display = 'block';
        gameSection.style.display = 'none';
    }
}

// --- GAME LOGIC VARIABLES ---
let quizData = [];
let currentQuestionIndex = 0;
let score = 0;

// 1. Fetch 20 random images from the backend
async function initGame() {
    document.getElementById('game-over-screen').style.display = 'none';
    document.getElementById('game-buttons').style.display = 'grid';
    document.getElementById('game-feedback').innerText = "Loading quiz...";
    document.getElementById('game-image').style.display = 'none';
    
    try {
        const response = await fetch('http://127.0.0.1:8000/api/game/start');
        const data = await response.json();
        quizData = data.quiz;
        currentQuestionIndex = 0;
        score = 0;
        
        loadQuestion();
    } catch (error) {
        document.getElementById('game-feedback').innerText = "Error loading game.";
        console.error(error);
    }
}

// 2. Display the current image
function loadQuestion() {
    document.getElementById('game-feedback').innerText = "";
    document.getElementById('game-progress').innerText = `Question ${currentQuestionIndex + 1} / 20`;
    document.getElementById('game-score').innerText = `Score: ${score}`;
    
    const imgElement = document.getElementById('game-image');
    imgElement.src = quizData[currentQuestionIndex].url;
    imgElement.style.display = 'inline-block';
    
    // Re-enable buttons
    const buttons = document.querySelectorAll('.k-btn');
    buttons.forEach(btn => btn.disabled = false);
}

// 3. Check the user's guess
function submitGuess(guessedLabel) {
    // Disable buttons so they can't double click
    const buttons = document.querySelectorAll('.k-btn');
    buttons.forEach(btn => btn.disabled = true);
    
    const actualLabel = quizData[currentQuestionIndex].label;
    const feedbackEl = document.getElementById('game-feedback');
    
    if (guessedLabel === actualLabel) {
        feedbackEl.innerText = "✅ Correct!";
        feedbackEl.style.color = "green";
        score++;
    } else {
        feedbackEl.innerText = `❌ Incorrect! It was ${actualLabel}`;
        feedbackEl.style.color = "red";
    }
    
    document.getElementById('game-score').innerText = `Score: ${score}`;
    
    // Wait 1.5 seconds so they can see the answer, then move to next
    setTimeout(() => {
        currentQuestionIndex++;
        if (currentQuestionIndex < quizData.length) {
            loadQuestion();
        } else {
            endGame();
        }
    }, 1500);
}

// 4. Show the final score
function endGame() {
    document.getElementById('game-image').style.display = 'none';
    document.getElementById('game-buttons').style.display = 'none';
    document.getElementById('game-feedback').innerText = "";
    document.getElementById('game-progress').innerText = "Quiz Complete!";
    
    document.getElementById('game-over-screen').style.display = 'block';
    
    const percentage = Math.round((score / 20) * 100);
    document.getElementById('final-score-text').innerText = `You scored ${score} / 20 (${percentage}%)`;
    
    // Compare their score to the AI!
    if (percentage >= 97) {
        document.getElementById('final-score-text').innerHTML += "<br>🤖 Wow! You tied the AI model!";
    } else {
        document.getElementById('final-score-text').innerHTML += "<br>📉 The AI scored 97%. Keep studying!";
    }
}