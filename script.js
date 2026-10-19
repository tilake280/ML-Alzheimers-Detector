// Switch between clinical and game views
function toggleMode(mode) {
    const clinicalSection = document.getElementById('clinical-section');
    const gameSection = document.getElementById('game-section');
    
    if (mode === 'game') {
        clinicalSection.style.display = 'none';
        gameSection.style.display = 'block';
        initGame();
    } else {
        clinicalSection.style.display = 'block';
        gameSection.style.display = 'none';
    }
}

// Game state
let quizData = [];
let currentQuestionIndex = 0;
let score = 0;

// Fetch 20 random images from the backend
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

// Show the current image
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

// Check the user's guess
function submitGuess(guessedLabel) {
    // Disable buttons to prevent double clicks
    const buttons = document.querySelectorAll('.k-btn');
    buttons.forEach(btn => btn.disabled = true);
    
    const actualLabel = quizData[currentQuestionIndex].label;
    const feedbackEl = document.getElementById('game-feedback');
    
    if (guessedLabel === actualLabel) {
        feedbackEl.innerText = "Correct!";
        feedbackEl.style.color = "green";
        score++;
    } else {
        feedbackEl.innerText = `Incorrect. It was ${actualLabel}`;
        feedbackEl.style.color = "red";
    }
    
    document.getElementById('game-score').innerText = `Score: ${score}`;
    
    // Show the answer for 1.5 seconds, then move on
    setTimeout(() => {
        currentQuestionIndex++;
        if (currentQuestionIndex < quizData.length) {
            loadQuestion();
        } else {
            endGame();
        }
    }, 1500);
}

// Show the final score
function endGame() {
    document.getElementById('game-image').style.display = 'none';
    document.getElementById('game-buttons').style.display = 'none';
    document.getElementById('game-feedback').innerText = "";
    document.getElementById('game-progress').innerText = "Quiz Complete!";
    
    document.getElementById('game-over-screen').style.display = 'block';
    
    const percentage = Math.round((score / 20) * 100);
    document.getElementById('final-score-text').innerText = `You scored ${score} / 20 (${percentage}%)`;
    
    // Compare against the model's accuracy
    if (percentage >= 97) {
        document.getElementById('final-score-text').innerHTML += "<br>You matched the model.";
    } else {
        document.getElementById('final-score-text').innerHTML += "<br>The model scored 97%.";
    }
}