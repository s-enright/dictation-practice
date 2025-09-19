document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element References ---
    const sentenceToDictateEl   = document.getElementById('sentenceToDictate');
    const newSentenceBtn        = document.getElementById('newSentenceBtn');
    const recordBtn             = document.getElementById('recordBtn');
    const recordingStatusEl     = document.getElementById('recordingStatus');
    const transcriptionOutputEl = document.getElementById('transcriptionOutput');
    const playCorrectBtn        = document.getElementById('playCorrectBtn');
    const audioPlayerEl         = document.getElementById('audioPlayer');
    const languageSelectorEl    = document.getElementById('languageSelector');
    const setLanguageBtn        = document.getElementById('setLanguageBtn');

    // --- State Variables ---
    let mediaRecorder;
    let audioChunks = [];
    let currentSentence = "";

    // --- Event Listeners ---
    newSentenceBtn.addEventListener('click', fetchNewSentence);
    recordBtn.addEventListener('click', toggleRecording);
    playCorrectBtn.addEventListener('click', playCorrectPronunciation);
    setLanguageBtn.addEventListener('click', setLanguage);

    // --- Core Functions ---
    /**
     * Gets a new sentence string from the get_sentence Flask endpoint and displays it,
     * clearing previous transcription data in the process.
     * @returns {Promise<void>}
     */
    async function fetchNewSentence() {
        try {
            const response = await fetch('/get_sentence'); // Calls the Flask endpoint
            if (!response.ok) {
                console.log(`HTTP error! Status: ${response.status}`);
                return;
            }
            const data = await response.json();
            currentSentence = data.sentence;
            sentenceToDictateEl.textContent = currentSentence;
            transcriptionOutputEl.value = ''; // Clear previous transcription
            playCorrectBtn.disabled = false; // Enable TTS button
            audioPlayerEl.classList.add('d-none'); // Hide audio player
            audioPlayerEl.src = ''; // Clear previous audio
        } catch (error) {
            console.error("Could not fetch new sentence:", error);
            sentenceToDictateEl.textContent = "Error fetching sentence. Please try again.";
        }
    }

    async function setLanguage() {
        const selectedLanguage = languageSelectorEl.value;
        try {
            const response = await fetch('/set_language', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ language: selectedLanguage })
            });
            if (response.ok) {
                const data = await response.json();
                // Display the sentence returned from the language change
                if (data.sentence) {
                    currentSentence = data.sentence;
                    sentenceToDictateEl.textContent = currentSentence;
                    transcriptionOutputEl.value = ''; // Clear previous transcription
                    playCorrectBtn.disabled = false; // Enable TTS button
                    audioPlayerEl.classList.add('d-none'); // Hide audio player
                    audioPlayerEl.src = ''; // Clear previous audio
                }
                console.log(data.message); // Log the success message
            } else {
                console.error('Failed to set language');
            }
        } catch (error) {
            console.error('Error setting language:', error);
        }
    }

    async function toggleRecording() {
        if (mediaRecorder && mediaRecorder.state === "recording") {
            // Stop recording
            mediaRecorder.stop();
            recordBtn.textContent = 'Start Recording';
            recordBtn.classList.remove('btn-danger', 'recording');
            recordBtn.classList.add('btn-success');
            recordingStatusEl.textContent = 'Processing...';
        } else {
            // Start recording
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = []; // Reset chunks

                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    // type 'audio/ogg; codecs=opus' or 'audio/webm; codecs=opus' might also work
                    // but wav is often well-supported by ASR backends.
                    // The backend will handle potential conversion if needed.

                    const formData = new FormData();
                    formData.append('audio_data', audioBlob, 'recording.wav');

                    try {
                        const response = await fetch('/process_audio', {
                            method: 'POST',
                            body: formData
                        });
                        if (!response.ok) {
                            console.log(`HTTP error! Status: ${response.status}`);
                            return;
                        }
                        const data = await response.json();
                        transcriptionOutputEl.value = data.transcription;
                        recordingStatusEl.textContent = 'Transcription complete!';
                    } catch (error) {
                        console.error("Error processing audio:", error);
                        transcriptionOutputEl.value = `Error: ${error.message}`;
                        recordingStatusEl.textContent = 'Error processing audio.';
                    }
                };

                mediaRecorder.start();
                recordBtn.textContent = 'Stop Recording';
                recordBtn.classList.remove('btn-success');
                recordBtn.classList.add('btn-danger', 'recording');
                recordingStatusEl.textContent = 'Recording...';
                transcriptionOutputEl.value = ''; // Clear previous transcription

            } catch (error) {
                console.error("Error accessing microphone:", error);
                recordingStatusEl.textContent = 'Could not access microphone.';
                alert("Could not access microphone. Please ensure permission is granted and try again.");
            }
        }
    }

    async function playCorrectPronunciation() {
        if (!currentSentence) {
            alert("Please get a sentence first.");
            return;
        }
        playCorrectBtn.disabled = true; // Prevent multiple clicks while processing
        playCorrectBtn.textContent = "Generating...";

        try {
            const response = await fetch('/synthesize_speech', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: currentSentence })
            });

            if (!response.ok) {
                console.log(`HTTP error! Status: ${response.status}`);
                return;
            }
            const data = await response.json();
            audioPlayerEl.src = data.audio_url; // Flask serves this from static/temp_audio
            audioPlayerEl.classList.remove('d-none'); // Show player
            audioPlayerEl.play();
        } catch (error) {
            console.error("Error synthesizing speech:", error);
            alert(`Could not synthesize speech: ${error.message}`);
        } finally {
            playCorrectBtn.disabled = false;
            playCorrectBtn.textContent = "Hear Correct Pronunciation";
        }
    }

    // --- Initialization ---
    // Optional: Fetch a sentence when the page loads
    // fetchNewSentence(); 
    // Or, let the user click the button to start.
});