

//uncomment begin experiment

//fix bonus
    const experimentStartTime = new Date();
    let currentPhase = 0;  //change to 1 to start from experiment
    let notesContainer = document.getElementById('notes');
    let InstructionnotesContainer = document.getElementById('notesInstruction');
    let instructionFeedback = document.getElementById('instructionFeedback');
    let experiment = document.getElementById('experiment');
    let feedbackContainer = document.getElementById('feedback');
    let birdContainer = document.getElementById('birdreply');
    let birdFeedback = document.getElementById('birdfeedback');
    let feedbackInstruction = document.getElementById('feedbackInstruction');
    // let timerContainer = document.getElementById('timer');
    let timerInstruction = document.getElementById('timerInstruction');
    let buttonContainer = document.getElementById('buttonContainer');
    let buttonInstructionContainer = document.getElementById('buttonInstructionContainer');
    let currentNoteIndex = 0;
    let phaseMelodies = [];
    let currentMelodyIndex = 0;
    let expectingUserInput = false;
    let experimentEnded = false;
    let score = 0;
    let pointsPerNote;
    let startPhaseTime;
    let endPhaseTime;
    let startTime;
    let endTime;
    let roundShifts = []
    let instructiondemo = false;

    let birdIntrophase3 = true;
    let birdinstruction = false;

    let totalscore = 0;  // to keep track of bonus;

    let roundBonus = [];





    let reward;   //change later

    // Get all melody boxes
    const melodyBoxes = document.querySelectorAll('.melody-box');

    let RoundtimePresses = [];
    let roundPressedKeys = [];

    let roundKeyPhase = {};
    let RoundTimePhase = {};
    let RoundPhaseScores = {};
    let note;
    let lineData;
    let noteElement;
    let pressedNotesStack = [];
    let notePrediction = 10;
    let resultPage = false;
    let totalMelodies = 9;
    let currentPage = 'experiment';
    let buttonDisabled = false;
    let totalPhaseTime = [];
    // let currentTimerInterval;
    let MelodyTotalTime = [];
    let roundPercentage = [];
    let roundScores = [];
    let line;
    let button;
    let spaceActive = false;
    let beginExperiment = false;
    let wrongNote = false;
    let key;
    let wrongnoteScore;
    let compFail = 0;
    let phasetransition = false;
    let insTransition = false;
    let sampled_indices = [];
    let roundMelodies = {};
    let birdPic = Array.from({ length: 5 }, () => Math.floor(Math.random() * 9) + 1);
    // console.log('birdPic', birdPic);
    for (let i = 0; i < 5; i++) {
        var Shift = Math.floor(Math.random() * 6);
        roundShifts.push(Shift);
    }
// console.log("shift",roundShifts)

    let sample_Melody = [[1, 2, 4, 3, 6, 5], [1, 2, 3, 4]]


    // let currentPage= 'Instruction8start';


    const birdImage = document.getElementById('bird-img');
    birdImage.src = `./src/images/bird${birdPic[currentPhase]}.png`;
    const phaseImg = document.getElementById('phase-img');
    phaseImg.src = `./src/images/bird${birdPic[currentPhase]}.png`;
    const birdReply = document.getElementById('birdreply-img');
    birdReply.src = `./src/images/bird${birdPic[currentPhase]}.png`;

    const birdResImage = document.getElementById('birdResult-img');
    birdResImage.src = `./src/images/bird${birdPic[currentPhase]}.png`;
    const MelodyImage = document.getElementById('MelodyBird');
    MelodyImage.src = `./src/images/bird${birdPic[currentPhase]}.png`;

    function getRandomIndices() {
        const indices = Array.from({ length: 104 }, (_, i) => i);
        while (sampled_indices.length < 5) {
            const ranIndex = Math.floor(Math.random() * indices.length);
            const selectedIndex = indices.splice(ranIndex, 1)[0];
            sampled_indices.push(selectedIndex);
        }
        // console.log(sampled_indices);
        return sampled_indices;
    }
    getRandomIndices()

    var fictionalcurrencies = ["&#976", "&#995", "&#985", "&#991", "&#993", "&#998", "&#1002", "&#1009", "&#999", "&#989", "&#983", "&#974", "&#8375;", "&#8379;", "&#9880;", "&#9799;", "&alefsym;", "&weierp;", "&#9797;", "&thetasym;", "&piv;", "&part;", "&#950;", "&#x37D;", "&#992;", "&#8713;", "&#164;", "&#186;", "&#926;", "&#8711;", "&#8855", "&#978;"];


    var fakecurrency = [];

    while (fakecurrency.length < 9) {
        var randomIndex = Math.floor(Math.random() * fictionalcurrencies.length);
        var currency = fictionalcurrencies[randomIndex];

        if (!fakecurrency.includes(currency)) {
            fakecurrency.push(currency);
        }
    }
  


    startPhaseTime = new Date();
    // console.log('startPhaseTime', startPhaseTime); //remove in end



    var fullurl = document.location.href, //url of incoming MTurk/prolific worker, used to extract workerid and assignmentid
        assignmentID,
        workerID,
        scenarioId,
        experimentData,
        gender = -1, //demographic info; update as needed
        age = -1;

    // extract URL parameters (FROM: https://s3.amazonaws.com/mturk-public/externalHIT_v1.js)
    function turkGetParam(name) {
        var regexS = "[\?&]" + name + "=([^&#]*)";
        var regex = new RegExp(regexS);
        var tmpURL = fullurl;
        var results = regex.exec(tmpURL);
        if (results == null) {
            return "";
        } else {
            return results[1];
        }
    }

    // CHANGE
    //Access the MySQL database and retrieve a scenario id, adding the now() date time to the start time field and marking the specific scenario as completed
    function assignScenario(callback) {
        var ajaxRequest = new XMLHttpRequest();
        try {
            // Opera 8.0+, Firefox, Safari
            ajaxRequest = new XMLHttpRequest();
        } catch (e) {
            // Internet Explorer Browsers
            try {
                ajaxRequest = new ActiveXObject("Msxml2.XMLHTTP");
            } catch (e) {
                try {
                    ajaxRequest = new ActiveXObject("Microsoft.XMLHTTP");
                } catch (e) {
                    // Something went wrong
                    alert("Your browser broke!");
                    return false;
                }
            }
        }
        var queryString = "?action=" + 'assignScenario';

        try {
            ajaxRequest.open("GET", "databasecall.php" + queryString, false); //The main functionality of this is defined in databasecall.php
            ajaxRequest.send();
            var response = ajaxRequest.responseText;

            var jsonArray = JSON.parse(response);
            scenarioId = parseInt(jsonArray['scenarioId']);
        } catch (e) {
            // Local preview without a PHP/MySQL backend: don't let a failed DB
            // call abort the rest of the script (which would leave pageSwitcher
            // undefined and break the consent button). Data saving won't work.
            console.warn('assignScenario failed (no backend?) - running in preview mode without data saving.', e);
            scenarioId = 0;
        }
        callback();
    }



    //Retrieve worker and assignment id from URL header, and then assigns them a scenario
    function begin_Experiment() { //called when participant grants consent
        // Retrieve assignmentID, workerID, ScenarioID, and environment from URL
        assignmentID = turkGetParam('STUDY_ID');

        workerID = turkGetParam('PROLIFIC_PID');

        // document.getElementById('PROLIFIC_PID').value = workerID; //prepopulate MTurk
        // CHANGE
        assignScenario(function () {

        });
    }


    begin_Experiment(); 

// sendData()
    const numToLetter = {
        1: "S",
        2: "D",
        3: "F",
        4: "J",
        5: "K",
        6: "L"
    }

    const noteToLineMap = {
        "S": "30",
        "D": "40",
        "F": "50",
        "J": "60",
        "K": "70",
        "L": "80"
    };






    window.instructioncheck = function () {
        //check if correct answers are provided
        if (document.getElementById('q1a').checked) {
            var ch1 = 1
        }
        if (document.getElementById('q2c').checked) {
            var ch2 = 1
        }
        if (document.getElementById('q3b').checked) {
            var ch3 = 1
        }


        //are all of the correct
        var checksum = ch1 + ch2 + ch3

        if (checksum === 3) {
            alert('Great, you have answered all of the questions correctly.');
            instructionCompleted = true;
            document.querySelector('#startExperiment img').src = `./src/images/bird${birdPic[currentPhase]}.png`;
            pageSwitcher('CompCheck', 'startExperiment');

            score = 0;


        }
        else {
            compFail++
            //if one or more answers are wrong, raise alert box
            alert('Incorrect. Please try again');
            //   pageSwitcher('compCheck', 'page2');
        }
    }
    // remove this

    window.pageSwitcher = function (hide, show) {
        document.getElementById(hide).style.display = 'none';
        document.getElementById(show).style.display = 'block';
        window.scrollTo(0, 0);


        if (show === 'Instruction1') {
            currentPage = 'Instruction1';
            const instruction = document.getElementById('Instruction1');
            instruction.style.display = 'flex';
        }
        if (show === 'Instruction2') {
            const instruction = document.getElementById('Instruction2');
            instruction.style.display = 'flex';
        }
        if (show === 'Instruction3') {
            currentPage = 'Instruction3';
        }
        if(show === 'CompCheck'){
            currentPage = 'CompCheck';   
        }


        if (show === 'startExperiment') {
            currentPage = 'startExperiment';
            currentMelodyIndex = 0;
            currentPhase = 0;
            score = 0;
            currentNoteIndex = 0;
            // console.log(roundPressedKeys,roundScores,RoundtimePresses,MelodyTotalTime)
            roundPressedKeys = [];
            roundScores = [];
            RoundtimePresses = [];
            roundPressedKeys = [];
            // console.log(roundPressedKeys,roundScores,RoundtimePresses,MelodyTotalTime)
            // console.log("dict",roundKeyPhase,RoundTimePhase,RoundPhaseScores)

            roundKeyPhase = {};
            RoundTimePhase = {};
            RoundPhaseScores = {};
            // console.log("dict",roundKeyPhase,RoundTimePhase,RoundPhaseScores)
            document.getElementById('score-value').textContent = 0

        }
        if (show === 'InstructionMelodyResult') {
            currentPage = 'InstructionMelodyResult';
        }
        if (show === 'Instruction7') {   
            currentPage = 'Instruction8';


        }
        if (show === 'Instruction11') {
            currentPage = 'Instruction11';


        }
        if (show === 'Instruction8') {
            currentPage = 'Instruction8start';
            document.getElementById('Part1ins').style.display = 'block';
            document.getElementById('Part2ins').style.display = 'none';
            document.getElementById('Part3ins').style.display = 'none';
            document.querySelector("#instructionDemo .part2instructions").style.display = "none";
            document.querySelector("#instructionDemo .part3instructions").style.display = "none";
            document.querySelector("#instructionDemo .part1instructions").style.display = "block";

            insTransition = false;
            clearNotes();

        }
        if (show === 'Instruction9') {
            currentPage = 'Instruction9start';
            clearNotes();
            insTransition = true;
            document.getElementById('Part1ins').style.display = 'none';
            document.getElementById('Part2ins').style.display = 'block';
            document.getElementById('Part3ins').style.display = 'none';
            document.querySelector("#instructionDemo .part1instructions").style.display = "none";
            document.querySelector("#instructionDemo .part3instructions").style.display = "none";
            document.querySelector("#instructionDemo .part2instructions").style.display = "block";
            
            document.getElementById.textContent = "Be as accurate as possible."
        }
        if (show === 'Instruction10') {
            clearNotes();
            insTransition = false;
            document.getElementById('Part1ins').style.display = 'none';
            document.getElementById('Part2ins').style.display = 'none';
            document.getElementById('Part3ins').style.display = 'block';

            document.querySelector("#instructionDemo .part1instructions").style.display = "none";
            document.querySelector("#instructionDemo .part2instructions").style.display = "none";
            document.querySelector("#instructionDemo .part3instructions").style.display = "block";

            currentPage = 'Instruction10start';
        }

        if (show === 'InstructionResult') {
            ContinueInstructionbuttonResult.disabled = false;
        }

        if (show === 'instructionDemo') {

            if (currentPage === 'Instruction8start') {
                document.getElementById('notesInsfeedback').style.display = 'none';
                expectingUserInput = false;
                instructiondemo = true;
                instructionFeedback.innerHTML = `Learning Time! <br> Press  <strong><b>SPACE</b></strong> to continue`;
                // console.log('calling experiment for instruction')
                return;
            }
            if (currentPage === 'Instruction9start') {


                if (insTransition) {
      
                    instructionFeedback.innerHTML = `Let's recap the song! <br> Press  <strong><b>SPACE</b></strong> to continue`;
                    document.getElementById('notesInsfeedback').style.display = 'none';


                    instructiondemo = true;
                    document.getElementById('InstructionbuttonContainer').style.display = 'none';
                    return;
                }
                else {
         
                    // document.getElementById('InstructionbuttonContainer').style.display = 'block';
                    document.getElementById('notesInsfeedback').style.display = 'block';
                    // document.getElementById('InstructionbuttonContainer').style.display = 'block';
                    instructionFeedback.innerHTML = "Do you remember?<br> Press  <strong><b>SPACE</b></strong> to continue";
                    document.getElementById('notesInsfeedback').textContent = `Notes: ${currentNoteIndex}/${sample_Melody[currentMelodyIndex].length}`;
                    instructiondemo = true;


                }
                const continueButton = document.querySelector(' #InstructionbuttonContainer .btn:nth-child(1)');
                continueButton.disabled = true;

                clearNotes();
                currentMelodyIndex = 0;
                currentNoteIndex = 0;




            }

            if (currentPage === 'Instruction10start') {
                instructionFeedback.innerHTML = `What will come next?<br> Press  <strong><b>SPACE</b></strong> to start!`;
                const continueButton = document.querySelector(' #InstructionbuttonContainer .btn:nth-child(1)');
                continueButton.disabled = true;

                clearNotes();
                currentMelodyIndex = 1;
                currentNoteIndex = 0;
                document.querySelector('#instructionDemo .part2instructions').style.display = 'none';
                document.querySelector('#instructionDemo .part3instructions').style.display = 'block';
                document.getElementById('notesInsfeedback').textContent = `Notes: ${currentNoteIndex}/${sample_Melody[currentMelodyIndex].length}`;
                // document.getElementById('InstructionbuttonContainer').style.display = 'block';
                document.getElementById('notesInsfeedback').style.display = 'block';
                instructiondemo = true;
                return;


            }
        }





        if (show === 'experiment') {
            clearNotes();

            if (currentPage === 'experimentPhase2' || currentPage === 'experimentPhase3') {
                document.getElementById('notesfeedback').textContent = `Notes: ${currentNoteIndex}/${phaseMelodies[currentMelodyIndex].length}`;
                document.getElementById('phase2transition').textContent = 'Make sure you are accurate.'
                // clearTimer()
                if (currentPage === 'experimentPhase3') {
                    const currentMelodyBox = melodyBoxes[currentPhase];
                    const subMelodies = currentMelodyBox.querySelectorAll('.submelody-box');
                    const currentSubmelody = subMelodies[currentMelodyIndex - 1];
                    const currentPhasebox = currentSubmelody.querySelectorAll('.phase-box')[2];
                    currentPhasebox.classList.add('active-phase');
                }

   
                // timerContainer.textContent = 60;
                expectingUserInput = false;
                beginExperiment = true;
                return;
            }
            else if (currentPage === 'experimentMelody') {
                document.getElementById('score').style.display = 'none';
                // document.getElementById('notesfeedback').textContent = `Notes: ${currentNoteIndex}/${phaseMelodies[currentMelodyIndex].length}`;
                // expectingUserInput = true;
                score = 0;
                resultPage = false;
                startPhaseTime = new Date();
                beginExperiment = true;
                return;
            } else {

                currentPage = 'experiment';
                // document.querySelector("#instructionDemo .part1instructions").style.display = "block";
                document.querySelector('#experiment .part1instructions').style.display = 'block';
                const currentMelodyBox = melodyBoxes[currentPhase];
                const subMelodies = currentMelodyBox.querySelectorAll('.submelody-box');
                const currentSubmelody = subMelodies[currentMelodyIndex];
                const currentPhasebox = currentSubmelody.querySelectorAll('.phase-box')[0];
                currentPhasebox.classList.add('active-phase');



                experiment.style.display = 'none';
                birdContainer.style.display = 'block';
                currentPage = 'birdintroFeedback'
                birdFeedback.innerHTML = `Melody ${currentPhase + 1} Part ${currentMelodyIndex + 1}.<br> Press <strong><b>SPACE</b></strong> to continue`;
                feedbackContainer.innerHTML = "Learning Time! <br> Press <strong><b>SPACE</b></strong> to continue";
                beginExperiment = true;
                document.getElementById('part1').style.display = 'block';
                expectingUserInput = false;
                document.getElementById('score').style.display = 'block';
                startPhaseTime = new Date();
            }



        }

        if (show === 'experimentResult') {
            resultPage = true;
            // ContinuebuttonResult.disabled = false;

            pressedNotesStack = [];
            roundPressedKeys = [];
            RoundtimePresses = [];
            clearNotes();
            // const continueButton = document.querySelector(' #buttonContainer .btn:nth-child(2)');
            // continueButton.disabled = true;

            expectingUserInput = false;
        }
        if (show === 'MelodyResult') {
            // ContinuebuttonResult.disabled = false;
        }
    }

    function clearNotes() {
        const allNotes = document.querySelectorAll('.note');
        allNotes.forEach(note => note.remove());
        currentNoteIndex = 0;

    }

    function loadMelodiesForPhase(phase) {
        let subPhase = 0;
        const maxSubPhase = 5;


        phaseMelodies = [];
        var shift = roundShifts[currentPhase];


   

        function fetchMelody(filePath) {
            return fetch(filePath)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`File not found: ${filePath}`);
                    }
                    return response.json();
                });
        }

        const melodyPromises = [];

        while (subPhase <= maxSubPhase) {
            const filePath = `training/${phase}-${subPhase}.json`;

            melodyPromises.push(fetchMelody(filePath));
            
            subPhase++;
        }

        Promise.all(melodyPromises)
            .then(results => {
                results = results.map(data => data.map(note => {
                    let shiftedNote = (note + shift) % 6;
                    return shiftedNote === 0 ? 6 : shiftedNote;
                }));
                results.forEach(data => phaseMelodies.push(data));
                roundMelodies[currentPhase] = phaseMelodies;

                
                
                startPhase();
                // console.log('phaseMelodies', phaseMelodies);
                subPhase += 1;
             
                document.getElementById('notesfeedback').textContent = ` Notes: 0/${phaseMelodies[currentMelodyIndex].length}`
            })
            .catch(error => {
      
                startPhase();
            });
    }

    function startPhase() {
        // console.log('start phase');
        if (phaseMelodies.length > 0) {
       
            pointsPerNote = 100 / (phaseMelodies[currentMelodyIndex].length);  // -1 to not include the first note in score calculation
    
            // currentMelodyIndex = 0;
            currentNoteIndex = 0;

            setTimeout(() => {
                displayNextNote();
                startTime = new Date();
            }, 500);

        } else {
            console.error('No melodies loaded for this phase.');
            endExperiment();
        }
    }


    function displayMelody() {

        feedbackContainer.textContent = `Let's recap the song!`;
  
        const melody = phaseMelodies[currentMelodyIndex];

        let delay = 0;
        const noteDuration = 700; // Adjust as needed; this is the delay between each note

        melody.forEach((note, index) => {
            setTimeout(() => {
 
                const experimentSection = document.getElementById('experiment');
  
                const button = experimentSection.querySelector(`.key[data-note="${numToLetter[note]}"]`);
       
                if (button) {
                    button.classList.add('clicked');
                    handleButtonClick({ target: button }, true);  // Pass true to signal this is from displayMelody
                    document.getElementById('notesfeedback').textContent = `Notes: ${currentNoteIndex}/${phaseMelodies[currentMelodyIndex].length}`;
                    setTimeout(() => {
                        button.classList.remove('clicked');
                    }, 200);  // This is the duration for how long the button stays clicked
                }
            }, delay);

            delay += noteDuration; // Increment the delay for each note
        });

        // Call pageSwitcher after the entire melody has been displayed
        setTimeout(() => {


            pageSwitcher('experiment', 'Phase2interruption');
        }, delay);  // Use the total accumulated delay for pageSwitcher
    }



    function displayInsMelody() {

        instructionFeedback.textContent = `Let's recap the song!`;
        expectingUserInput = false;

        const melody = sample_Melody[currentMelodyIndex];
  
        let delay = 0;
        const noteDuration = 700; // Adjust as needed; this is the delay between each note

        melody.forEach((note, index) => {
            setTimeout(() => {
                // console.log(melody);
                const instructionDemo = document.getElementById('instructionDemo');
                // console.log(note)
                const button = instructionDemo.querySelector(`.key[data-note="${numToLetter[note]}"]`);

                // console.log('button', button);
                if (button) {
                    button.classList.add('clicked');
                    handleButtonClick({ target: button }, true);  // Pass true to signal this is from displayMelody
                    setTimeout(() => {
                        button.classList.remove('clicked');
                    }, 200);  // This is the duration for how long the button stays clicked
                }
            }, delay);

            delay += noteDuration; // Increment the delay for each note
        });

        // Call pageSwitcher after the entire melody has been displayed
        setTimeout(() => {
            // console.log('Switching page after melody');
            instructionFeedback.textContent = `Now it's your turn!`;
            insTransition = false;
            pageSwitcher('instructionDemo', 'instructionDemo');
        }, delay);  // Use the total accumulated delay for pageSwitcher
    }



    function displayInsNote() {
        if (experimentEnded) return;
        instructionFeedback.textContent = `Press the notes highlighted in blue!`;
        spaceActive = false;
        let noteName = sample_Melody[currentMelodyIndex][currentNoteIndex];
        // console.log('noteName', noteName);
        // console.log('noteName', [numToLetter[noteName]]);
        let noteElement = document.createElement('div');
        noteElement.className = 'note';
        noteElement.style.top = '0px';
        noteElement.style.backgroundColor = 'darkblue';

        const lineData = noteToLineMap[numToLetter[noteName]];

        line = document.querySelector(`.line[data-line="${lineData}"]`);

        if (line) {
            // line.appendChild(noteElement);    // to make the note go down immediately
            const existingNotes = document.querySelectorAll('.note');
          
            const lineHeight = line.offsetHeight;

            existingNotes.forEach(existingNote => {
                let newTop = parseFloat(existingNote.style.top || '0px') + 80;

                if (newTop >= lineHeight - 20) {
           
                    existingNote.remove();
                } else {
                    existingNote.style.top = `${newTop}px`;

                    existingNote.style.backgroundColor = 'rgba(84, 157, 230, 0.429)';
                }
            });

            line.appendChild(noteElement);
            requestAnimationFrame(() => {
      
                noteElement.classList.add('move-down');
            });
            expectingUserInput = true;
            if (currentPage != 'Instruction8start') {
         
                currentNoteIndex++;
            }




            startTime = new Date();      //correct date

        } else {
            console.error(`No line found for data-line="${lineData}"`);
        }
    }


    function displayNextNote() {
        let noteName;
        if (experimentEnded) return;

        feedbackContainer.textContent = `Press the notes highlighted in blue!`;


        spaceActive = false;

        if (currentPage === 'experiment' || currentPage === 'Instruction8start') {


            if (currentPage === 'experiment') {
                if (currentNoteIndex >= phaseMelodies[currentMelodyIndex].length) {
                    endExperiment();
                    birdContainer.style.display = 'none';
                    return;

                }
                noteName = phaseMelodies[currentMelodyIndex][currentNoteIndex];

            }


            let noteElement = document.createElement('div');
            noteElement.className = 'note';
            noteElement.style.top = '0px';
            noteElement.style.backgroundColor = 'darkblue';
            const lineData = noteToLineMap[numToLetter[noteName]];


            // Assuming lineData is already defined and contains the value like '30', '40', etc.

            const experimentSection = document.getElementById('experiment');

            line = experimentSection.querySelector(`.line[data-line="${lineData}"]`);
   


            if (line) {
                // line.appendChild(noteElement);    // to make the note go down immediately
                const existingNotes = document.querySelectorAll('.note');
 
                const lineHeight = line.offsetHeight;

                existingNotes.forEach(existingNote => {
                    let newTop = parseFloat(existingNote.style.top || '0px') + 80;
              
                    if (newTop >= lineHeight - 20) {
                        existingNote.remove();
                    } else {
                        existingNote.style.top = `${newTop}px`;

                        existingNote.style.backgroundColor = 'rgba(84, 157, 230, 0.429)';
                    }
                });

                line.appendChild(noteElement);
                requestAnimationFrame(() => {
                   
                    noteElement.classList.add('move-down');
                });
                expectingUserInput = true;
             
                if (currentPage != 'experiment') {
                    currentNoteIndex++;
                }

                // console.log("currentNoteIndex", currentNoteIndex);
                // console.log("currentmelody index", currentMelodyIndex)
                // console.log("phaseMelodies[currentMelodyIndex]", phaseMelodies[currentMelodyIndex]);


                startTime = new Date();      //correct date

            } else {
                console.error(`No line found for data-line="${lineData}"`);
            }
        }
    }




    async function handleKeyPress(event) {
        // console.log('key press');
        // console.log("sp", spaceActive);



        event.preventDefault();






        // Existing logic for other keys or actions



        const validKeys = ['S', 'D', 'F', 'J', 'K', 'L'];
        let key;

        if (currentPage === 'birdintroFeedback' && (event.code === 'Space' || event.code === ' ')) {
            currentPage = 'experiment';
            birdContainer.style.display = 'none';
            // console.log('currentpage', currentPage);
            experiment.style.display = 'block';
            return;
        }

        if (instructiondemo && currentPage === 'Instruction8start' && (event.code === 'Space' || event.code === ' ')) {
            instructiondemo = false;
            expectingUserInput = false;
            displayInsNote();
            // console.log("displaying note")
            instructionFeedback.textContent = `Press the notes highlighted in blue!`;
        }
        if (instructiondemo && currentPage === 'Instruction9start' && (event.code === 'Space' || event.code === ' ')) {
            ;

            if (insTransition) {
                instructiondemo = false;
                expectingUserInput = false;
                // console.log('here')
                clearNotes();
                document.getElementById('notesInsfeedback').style.display = 'none';
                currentMelodyIndex = 0;
                currentNoteIndex = 0;
                displayInsMelody();
                return;
            } else {
                instructiondemo = false;
                expectingUserInput = true;
                instructionFeedback.textContent = `Play back the melody from memory!`;
            }
        }

        if (instructiondemo && currentPage === 'Instruction10start' && (event.code === 'Space' || event.code === ' ')) {
            ;
            instructiondemo = false;
            expectingUserInput = true;
            instructionFeedback.textContent = `What will come next?`;

        }

        // Handle different phases of the experiment
        if (beginExperiment && (currentPage === 'experiment' || currentPage === 'Instruction8start') && (event.code === 'Space' || event.code === ' ')) {

            // console.log('begin experiment after space');
            // console.log('currentPage', currentPage);
            await loadMelodiesForPhase(sampled_indices[currentPhase]);  // Await async loading of melodies

            if (phaseMelodies.length > 0) {

                document.getElementById('notesfeedback').textContent = `Notes: ${currentNoteIndex}/${phaseMelodies[currentMelodyIndex].length}`;

            }
            birdContainer.style.display = 'none';
            experiment.style.display = 'block';
            beginExperiment = false;
            expectingUserInput = false;
            return;
        }

        if (beginExperiment && currentPage === 'experimentMelody' && (event.code === 'Space' || event.code === ' ')) {
            // console.log('begin experiment after space');

            expectingUserInput = false;
            clearNotes();
            await displayMelody();  // Await async melody display
            beginExperiment = false;
            return;
        }

        if (beginExperiment && (currentPage === 'experimentPhase2' || currentPage === 'experimentPhase3') && (event.code === 'Space' || event.code === ' ')) {
            // console.log('begin experiment after space');
            beginExperiment = false;
            RoundtimePresses = [];
            roundPressedKeys = [];
            startTime = new Date();
            if (currentPage === 'experimentPhase2') {

                feedbackContainer.textContent = 'Play back the melody from memory!';
            } else if (currentPage === 'experimentPhase3') {

                feedbackContainer.textContent = 'What will come next?';
            }

            expectingUserInput = true;
            // clearTimer();
            // clearInterval(window.currentTimerInterval);
            // window.currentTimerInterval = startTimer(60);  // Start timer
            document.getElementById('notesfeedback').style.display = 'block';
            if (currentPage === 'experimentPhase2') {
                document.getElementById('notesfeedback').textContent = ` Notes: 0/${phaseMelodies[currentMelodyIndex].length}`;
            } else if (currentPage === 'experimentPhase3') {
                document.getElementById('notesfeedback').textContent = ` Notes: 0/${phaseMelodies[currentMelodyIndex].length}`;
            }
            return;
        }

        if ((event.code === 'Space' || event.code === ' ') && spaceActive && (currentPage === 'experiment' || (currentPage === 'Instruction8start'))) {
            // console.log('space key pressed');
            // console.log('currentPage', currentPage);
            // console.log('spaceActive', spaceActive);
            // mouseEventsEnabled = false;  // Disable mouse events during space handling
            key = null;
            // await handleSpace();  // Await space handling if there are async operations involved

            expectingUserInput = true;
            startTime = new Date();
            // setTimeout(() => {
            //     expectingUserInput = true;
            // }, 110);
            return;
        }


        // Check if the event is a keyboard press
        if (event.type === 'keydown') {
            key = event.key.toUpperCase();  // Get the key from the keyboard
            // console.log('Key pressed:', key);
        }


        if (validKeys.includes(key)) {
            let button;
            if (currentPage === 'Instruction8start' || currentPage === 'Instruction9start' || currentPage === 'Instruction10start') {
                button = document.querySelector(`.key[data-note="${key}"]`);
            } else {
                const experimentSection = document.getElementById('experiment');
                button = experimentSection.querySelector(`.key[data-note="${key}"]`);
            }

            if (button) {
                // console.log('button clicked', button);
                await handleButtonClick({ target: button });  // Await button handling if it's async

                // console.log("sucessssssss")
                return;
            }
        }





        if (experimentEnded || !expectingUserInput) return;


    }



    // // Add keydown event listener for the space key
    document.addEventListener('keydown', handleKeyPress);

    // Get the continue buttons
    const ContinuebuttonResult = document.getElementById('buttonResult');
    const experimentResult = document.getElementById('experimentResult');
    // console.log('ContinuebuttonResult', ContinuebuttonResult);
    const ContinueInstructionbuttonResult = document.getElementById('InstructionbuttonResults');
    const MelodybuttonResult = document.getElementById('buttonMelodyResult');
    const MelodyResult = document.getElementById('MelodyResult');

    const InstructionMelodybuttonResult = document.getElementById('InstructionbuttonMelodyResult');
    const Phase2interruption = document.getElementById('Phase2interruption');
    const birdreply = document.getElementById('birdreply');
    const Ins7Button = document.getElementById('Ins7Button');
    const Ins8Button = document.getElementById('Ins8Button');
    const Ins9Button = document.getElementById('Ins9Button');
    const Ins10Button = document.getElementById('Ins10Button');
    const melodyChange = document.getElementById('Melodychange');




    // Add click event listeners if the buttons exist
    if (ContinuebuttonResult) {
        ContinuebuttonResult.addEventListener('click', (event) => {
            event.preventDefault();
            handleContinueButtonClick(event);
        });
    }
    if (ContinueInstructionbuttonResult) {
        ContinueInstructionbuttonResult.addEventListener('click', (event) => {
            event.preventDefault();
            handleContinueButtonClick(event);
        });
    }
    if (MelodybuttonResult) {
        MelodybuttonResult.addEventListener('click', handleContinueButtonClick);
        // console.log('Melody next button');
    }
    if (InstructionMelodybuttonResult) {
        InstructionMelodybuttonResult.addEventListener('click', handleContinueButtonClick);
        // console.log('Melody next button');
    }


    // Add keydown event listener for space key
    document.addEventListener('keydown', (event) => {
        event.preventDefault()

        if (event.key === ' ') {
            if (currentPage === 'startExperiment') {
                pageSwitcher('startExperiment', 'experiment');
                feedbackContainer.innerHTML = `Learning Time! <br> Press  <strong><b>SPACE</b></strong> to continue`;
                event.key = ''
                return;
            }
            // console.log('space click')
            // Check if any of the continue buttons are visible


            const isButtonResultVisible = ContinuebuttonResult && ContinuebuttonResult.offsetWidth > 0 && ContinuebuttonResult.offsetHeight > 0;
            const experimetnResultvisible = experimentResult && experimentResult.offsetWidth > 0 && experimentResult.offsetHeight > 0;
            const MelodyResultvisible = MelodyResult && MelodyResult.offsetWidth > 0 && MelodyResult.offsetHeight > 0;
            const isInstructionButtonResultVisible = ContinueInstructionbuttonResult && ContinueInstructionbuttonResult.offsetWidth > 0 && ContinueInstructionbuttonResult.offsetHeight > 0;
            const isMelodyButtonResultVisible = MelodybuttonResult && MelodybuttonResult.offsetWidth > 0 && MelodybuttonResult.offsetHeight > 0;
            const isInstructionMelodyButtonResultVisible = InstructionMelodybuttonResult && InstructionMelodybuttonResult.offsetWidth > 0 && InstructionMelodybuttonResult.offsetHeight > 0;
            const isphase2visible = Phase2interruption && Phase2interruption.offsetWidth > 0 && Phase2interruption.offsetHeight > 0;
            const melodyChangevisible = melodyChange && melodyChange.offsetWidth > 0 && melodyChange.offsetHeight > 0;
            const birdreplyvisible = birdreply && birdreply.offsetWidth > 0 && birdreply.offsetHeight > 0;
            // console.log('isphase2visible', isphase2visible);
            const ins7visible = Ins7Button.offsetWidth > 0 && Ins7Button.offsetHeight > 0;
            const ins8visible = Ins8Button.offsetWidth > 0 && Ins8Button.offsetHeight > 0;
            const ins9visible = Ins9Button.offsetWidth > 0 && Ins9Button.offsetHeight > 0;
            const ins10visible = Ins10Button.offsetWidth > 0 && Ins10Button.offsetHeight > 0;

            const Ins1 = document.getElementById('Instruction1');
            const Ins2 = document.getElementById('birdsInstruction');
            const Ins3 = document.getElementById('Instruction3');
            const Ins4 = document.getElementById('Instruction4');
            const Ins5 = document.getElementById('Instruction5');
            const Ins6 = document.getElementById('Instruction6');
            const Ins11 = document.getElementById('Instruction11');

            if (isphase2visible) {
                currentPage = 'phase2block';
                // console.log('phase2block phase 2visibke')
                clearNotes()
                handleContinueButtonClick(event);
                return;
            }


            if (Ins1 && Ins1.offsetWidth > 0 && event.code === 'Space' || event.code === ' ') {
                // console.log('here')
                pageSwitcher('Instruction1', 'birdsInstruction');
                return;
            }
            if (Ins2 && Ins2.offsetWidth > 0 && event.code === 'Space' || event.code === ' ') {
                pageSwitcher('birdsInstruction', 'Instruction3');
                return;
            }
            if (Ins3 && Ins3.offsetWidth > 0 && event.code === 'Space' || event.code === ' ') {
                pageSwitcher('Instruction3', 'Instruction4');
                return;
            }
            if (Ins4 && Ins4.offsetWidth > 0 && event.code === 'Space' || event.code === ' ') {
                pageSwitcher('Instruction4', 'Instruction5');
                return;
            }
            if (Ins5 && Ins5.offsetWidth > 0 && event.code === 'Space' || event.code === ' ') {
                pageSwitcher('Instruction5', 'Instruction6');
                return;
            }
           
            if (Ins6 && Ins6.offsetWidth > 0 && event.code === 'Space' || event.code === ' ') {
                pageSwitcher('Instruction6', 'Instruction7');
                return;
            }

            if (ins7visible) {
                pageSwitcher('Instruction7', 'Instruction8');
                return;
            }
            else if (ins8visible) {
                pageSwitcher('Instruction8', 'instructionDemo');

                return;
            }
            else if (ins9visible) {
                removeKeyListeners();
                clearNotes();
                pageSwitcher('Instruction9', 'instructionDemo');

                // console.log('instruction9switching')
                addKeyListeners();
                return;
            }
            else if (ins10visible) {
                removeKeyListeners();
                pageSwitcher('Instruction10', 'instructionDemo');
                // console.log('instruction10switching')
                addKeyListeners();
                return;
            }
            if (Ins11 && Ins11.offsetWidth > 0 && event.code === 'Space' || event.code === ' ') {
                pageSwitcher('Instruction11', 'CompCheck');
                return;
            }
            if (melodyChangevisible) {
                pageSwitcher('Melodychange', 'experiment');
                return;
            }
            // If any button is visible, trigger the click handler
            else if (isButtonResultVisible || isInstructionButtonResultVisible || isMelodyButtonResultVisible || isInstructionMelodyButtonResultVisible || birdreplyvisible || experimetnResultvisible || MelodyResultvisible) {
                if (!birdinstruction && birdreplyvisible) {
                    // console.log("currentPage", currentPage);
                    if (currentPage === 'experiment') {
                        return;
                    }
                }
                // console.log('button click continue')
                handleContinueButtonClick(event);
                return;
            }

        }
    });

    function pushRoundkeys(noteIndex, noteValue) {
        // Check if the noteIndex already exists in the stack
        if (!roundPressedKeys[noteIndex]) {
            // If no entry exists, add the note directly
            roundPressedKeys[noteIndex] = noteValue;
        } else if (Array.isArray(roundPressedKeys[noteIndex])) {
            // If a nested list already exists, append the new note
            roundPressedKeys[noteIndex].push(noteValue);
        } else {
            // If a single value exists, convert it into a nested list and append the new note
            roundPressedKeys[noteIndex] = [roundPressedKeys[noteIndex], noteValue];
        }

    }
    function pushTime(noteIndex, time) {
        // Check if the noteIndex already exists in the stack
        if (!RoundtimePresses[noteIndex]) {
            // If no entry exists, add the note directly
            RoundtimePresses[noteIndex] = time;
        } else if (Array.isArray(RoundtimePresses[noteIndex])) {
            // If a nested list already exists, append the new note
            RoundtimePresses[noteIndex].push(time);
        } else {
            // If a single value exists, convert it into a nested list and append the new note
            RoundtimePresses[noteIndex] = [RoundtimePresses[noteIndex], time];
        }
    }

    async function handleButtonClick(event, fromDisplayMelody = false) {
        // Only check expectingUserInput if it's not from displayMelody
        // console.log('expectingUserInput', expectingUserInput);
        // console.log('button click removed');
        if (spaceActive) {
            return;
        }

        if (!fromDisplayMelody && (!expectingUserInput || experimentEnded)) {
            // console.log('heehhhe')
            event.target = null;
            // console.log(spaceActive);
            if (currentPage === 'experiment' && spaceActive) {
                removeKeyListeners();
                // console.log('space active')
                addKeyListeners();
                return;
            }
            // console.log('button click not allowed');

            return;
        }

        expectingUserInput = false;  // You may want to modify this based on your use case.
        // console.log('c', currentPage);









        if (currentPage === 'experiment' && currentNoteIndex >= phaseMelodies[currentMelodyIndex].length) {
            // console.log('hhhh')
            expectingUserInput = false;
            return;
        }




        endTime = new Date() - startTime;
        // RoundtimePresses.push(endTime);
        note = event.target.dataset.note;
        pushRoundkeys(currentNoteIndex, note);
        pushTime(currentNoteIndex, endTime);

        // roundPressedKeys.push(note);
        // console.log("RoundtimePresses", RoundtimePresses)
        // console.log("roundPressedkeys", roundPressedKeys)

        lineData = event.target.dataset.line;

        let noteElement = document.createElement('div');
        noteElement.className = 'note';

        noteElement.style.left = event.target.style.left;
        noteElement.style.top = '0px';
        if (currentPage === 'Instruction9start' || currentPage === 'Instruction10start' || currentPage === 'experimentPhase2' || currentPage === 'experimentPhase3' || currentPage === 'experimentMelody') {
            noteElement.style.backgroundColor = 'darkblue';
        }

        if (currentPage === 'Instruction8start' || currentPage === 'Instruction9start' || currentPage === 'Instruction10start') {
            line = document.querySelector(`.line[data-line="${lineData}"]`);
            // console.log('line', line);
        } else {
            const experimentSection = document.getElementById('experiment');
            // console.log('experimentSection', experimentSection);
            line = experimentSection.querySelector(`.line[data-line="${lineData}"]`);
            // console.log('line', line);
        }

        if (line) {
            // console.log('notes moving down');
            const existingNotes = document.querySelectorAll('.note');
            // console.log("Existing notes", existingNotes);
            const lineHeight = line.offsetHeight;
            if (currentPage !== 'experiment' && currentPage !== 'Instruction8start') {
                // console.log(currentPage)
                existingNotes.forEach(existingNote => {
                    let newTop = parseFloat(existingNote.style.top || '0px') + 80;
                    if (newTop >= lineHeight - 20) {
                        existingNote.remove();
                    } else {
                        existingNote.style.top = `${newTop}px`;
                        existingNote.style.backgroundColor = 'rgba(84, 157, 230, 0.429)';
                    }
                });
            }

            line.appendChild(noteElement);
            requestAnimationFrame(() => {
                noteElement.classList.add('move-down');
            });

            // pushNoteToStack(currentNoteIndex,noteElement);
            pressedNotesStack.push(noteElement);
            // console.log('pressedNotesStack', pressedNotesStack);



            const button = event.target;

            event.target.classList.add('clicked');
            setTimeout(() => {
                event.target.classList.remove('clicked');
            }, 400);

            if (currentPage === 'experiment' || currentPage === 'Instruction8start') {
                
                // console.log('Recording')
                // console.log('event.target', event.target);

                recordUserInput(note, noteElement);
                // console.log(event);
             
                // console.log('note', event.target);

                // expectingUserInput = false;



            } else {
                recordUserInput(note, noteElement);
                // setTimeout(() => {
                //     event.target.classList.remove('clicked');
                // }, 400);

                if (currentPage === 'experimentPhase2' && currentNoteIndex >= phaseMelodies[currentMelodyIndex].length) {

                    // console.log("length", phaseMelodies[currentMelodyIndex].length)
                    const continueButton = document.querySelector(' #buttonContainer .btn:nth-child(1)');
                    event.target.classList.add('clicked');
                    // console.log(event);

                    // setTimeout(() => {
                    //     event.target.classList.remove('clicked');
                    // }, 400);
                    // console.log('note', event.target);

                    expectingUserInput = false;
                    continueButton.disabled = false;
                    setTimeout(() => {
                    
                            clearNotes()
                            endExperiment();
   
                    }, 1000);
                    // birdFeedback.innerHTML = 'Press <strong><b>SPACE</b></strong> to continue';
                    // feedbackContainer.innerHTML = 'Press <strong><b>SPACE</b></strong> to continue';
                    // console.log("more notes")
                    return;
                }
                if (currentPage === 'experimentPhase3' && currentNoteIndex >= phaseMelodies[currentMelodyIndex].length) {
                    noteElement.style.backgroundColor = 'darkblue';
                    // console.log("length", phaseMelodies[currentMelodyIndex].length)
                    const continueButton = document.querySelector(' #buttonContainer .btn:nth-child(1)');
                    event.target.classList.add('clicked');

                    // setTimeout(() => {
                    //     event.target.classList.remove('clicked');
                    // }, 400);
                    // console.log('note', event.target);

                    expectingUserInput = false;
                    // continueButton.disabled = false;
                    setTimeout(() => {
                    
                        clearNotes()
                        endExperiment();

                }, 1000);
                    birdFeedback.innerHTML = 'Press <strong><b>SPACE</b></strong> to continue';
                    // feedbackContainer.innerHTML = 'Press <strong><b>SPACE</b></strong> to continue';
                    // console.log("more notes")
                    return;
                }

                if (currentPage === 'Instruction9start' && currentNoteIndex >= sample_Melody[currentMelodyIndex].length) {

                    // console.log("length", sample_Melody[currentMelodyIndex].length)
                    const continueButton = document.querySelector(' #InstructionbuttonContainer .btn:nth-child(1)');
                    event.target.classList.add('clicked');

                    // setTimeout(() => {
                    //     event.target.classList.remove('clicked');
                    // }, 400);
                    // console.log('note', event.target);

                    expectingUserInput = false;
                    continueButton.disabled = false;

                    // instructionFeedback.innerHTML = 'Press <strong><b>SPACE</b></strong> to continue';
                    // console.log(continueButton)
                    // console.log("more notes")
                    return;
                }

                if (currentPage === 'Instruction10start' && currentNoteIndex >= sample_Melody[currentMelodyIndex].length) {

                    const continueButton = document.querySelector(' #InstructionbuttonContainer .btn:nth-child(1)');
                    event.target.classList.add('clicked');
                    // setTimeout(() => {
                    //     event.target.classList.remove('clicked');
                    // }, 400);
                    // console.log('note', event.target);

                    expectingUserInput = false;
                    continueButton.disabled = false;
                    // instructionFeedback.innerHTML = 'Press <strong><b>SPACE</b></strong> to continue';
                    return;
                }

            }

        } else {
            console.error(`No line found for data-line="${lineData}"`);
        }
    }


    function handleContinueButtonClick() {
        // console.log('continue button clicked');
        // console.log('current page', currentPage);
        if (currentPage === 'phase2block') {
            document.removeEventListener('keydown', handleKeyPress);
            // console.log("phase2block");
            feedbackContainer.innerHTML = "Do you remember?";
            feedbackContainer.innerHTML += "<br>";
            feedbackContainer.innerHTML += "Press  <strong><b>SPACE</b></strong> to start";
            document.getElementById('notesfeedback').textContent = `Notes: ${currentNoteIndex}/${phaseMelodies[currentMelodyIndex].length}`;
            phasetransition = false    //since it gets activated during display melody

            // console.log('current page', currentPage);
            let currentMelodyBox = melodyBoxes[currentPhase];
            let subMelodies = currentMelodyBox.querySelectorAll('.submelody-box');
            let currentSubmelody = subMelodies[currentMelodyIndex];
            let currentPhasebox = currentSubmelody.querySelectorAll('.phase-box')[0];
            currentPhasebox.classList.remove('active-phase');
            currentPhasebox.classList.add('completed-phase');


            currentPage = 'experimentPhase2';
            // console.log('current page', currentPage);

            roundPercentage.push(score);
            // console.log("round percentage", roundPercentage)
            // console.log("roundScores", roundScores);
            // timerContainer.style.display = 'block';
            // buttonContainer.style.display = 'block';
            const continueButton = document.querySelector(' #buttonContainer .btn:nth-child(1)');
            continueButton.disabled = true;

            beginExperiment = false;
            expectingUserInput = false;

            pageSwitcher('Phase2interruption', 'experiment');
            document.addEventListener('keydown', handleKeyPress);
            return;
        }
        else if (currentPage === 'experimentPhase2') {
            // console.log("currentMeldoyindex", currentMelodyIndex)
            // Change the bird image based on the current phase number

            if (phasetransition && currentMelodyIndex == 4) {
                document.removeEventListener('keydown', handleKeyPress);   // to remove space from activating timer

                // console.log("5th submelody")
                const currentMelodyBox = melodyBoxes[currentPhase];
                const subMelodies = currentMelodyBox.querySelectorAll('.submelody-box');
                const currentSubmelody = subMelodies[currentMelodyIndex];
                const currentPhasebox = currentSubmelody.querySelectorAll('.phase-box')[1];
                currentPhasebox.classList.remove('active-phase');
                currentPhasebox.classList.add('completed-phase');

                // document.getElementById('scoreFeedback').textContent = 'Welcome to learning phase 3'
                feedbackContainer.innerHTML = "What will come next?<br> Press  <strong><b>SPACE</b></strong> to start!";
                // console.log("score",score);
                score = 0;
                // console.log("score",score);
                phasetransition = false
                document.getElementById('part3').style.display = 'block';
                currentMelodyIndex++;
                // console.log("currentMelodyIndex", currentMelodyIndex);
                // console.log('phaseMelodies[currentMelodyIndex]', phaseMelodies[currentMelodyIndex])
                document.getElementById('notesfeedback').textContent = `Notes: 0/${phaseMelodies[currentMelodyIndex].length}`;
                document.querySelector('#experiment .part3instructions').style.display = 'block';
                document.querySelector('#experiment .part2instructions').style.display = 'none';
                document.querySelector('#experiment .part1instructions').style.display = 'none';
                document.getElementById('part2').style.display = 'none';
                // timerContainer.style.display = 'block';
                // buttonContainer.style.display = 'block';
                currentPage = 'experimentPhase3';

                expectingUserInput = false;
                startTime = new Date();
                const continueButton = document.querySelector(' #buttonContainer .btn:nth-child(1)');
                continueButton.disabled = true;

                pageSwitcher('experimentResult', 'experiment')
                document.addEventListener('keydown', handleKeyPress);

                // pageSwitcher('experimentResult', 'birdreply')

                // clearTimer();
                // clearInterval(window.currentTimerInterval);
                // console.log("current page 3", currentPage);
                return
            } else {
                phasetransition = false
                // console.log('page 2')

                let percent = score;
                // console.log("percent", percent)
                roundPercentage.push(percent);
                // console.log("score", score);
                score = 0;
                // console.log("phasemelodies", phaseMelodies);
                // console.log("part2", phaseMelodies[currentMelodyIndex]);
                // console.log("currentMelodyIndex", currentMelodyIndex);

                const currentMelodyBox = melodyBoxes[currentPhase];
                const subMelodies = currentMelodyBox.querySelectorAll('.submelody-box');
                const currentSubmelody = subMelodies[currentMelodyIndex];
                const currentPhasebox = currentSubmelody.querySelectorAll('.phase-box')[1];
                currentPhasebox.classList.remove('active-phase');
                currentPhasebox.classList.add('completed-phase');

                currentMelodyIndex++;

                // console.log("currentMelodyIndex", currentMelodyIndex);
                // console.log("part3", phaseMelodies[currentMelodyIndex]);
                // feedbackContainer.textContent = `Predict the next melody?!!!!!`;
                // timerContainer.style.display = 'none';
                // buttonContainer.style.display = 'none';
                const continueButton = document.querySelector(' #buttonContainer .btn:nth-child(1)');
                document.querySelector('#experiment .part2instructions').style.display = 'none';
                document.querySelector('.part1instructions').style.display = 'block';
                document.getElementById('part2').style.display = 'none';

                document.getElementById('score').style.display = 'none';
                document.getElementById('score-value').textContent = score;


                feedbackContainer.innerHTML = "Learning time! <br> Press <strong><b>SPACE</b></strong> to continue";



                expectingUserInput = false;
                continueButton.disabled = true;



                currentPage = 'MelodyResult';
                document.getElementById('notesfeedback').display = 'block';
                document.getElementById('notesfeedback').textContent = `Notes: 0/${phaseMelodies[currentMelodyIndex].length}`;
                birdinstruction = false;
                pageSwitcher('experimentResult', 'experiment');
                document.getElementById('score-value').textContent = score;
                // console.log("current page 3", currentPage);
                // ContinuebuttonResult.disabled = true;

                return;
            }


        } else if (currentPage === 'experiment') {
            if (phasetransition) {
                expectingUserInput = false;
                // console.log("phase 2")

                const currentMelodyBox = melodyBoxes[currentPhase];
                const subMelodies = currentMelodyBox.querySelectorAll('.submelody-box');
                const currentSubmelody = subMelodies[currentMelodyIndex];
                let currentPhasebox = currentSubmelody.querySelectorAll('.phase-box')[0];
                currentPhasebox.classList.remove('active-phase');
                currentPhasebox.classList.add('completed-phase');

                currentPhasebox = currentSubmelody.querySelectorAll('.phase-box')[1];
                currentPhasebox.classList.add('active-phase');
                document.getElementById('scoreFeedback').textContent = 'Welcome to learning phase 2'

       
                document.removeEventListener('keydown', handleKeyPress);

                feedbackContainer.innerHTML = "Let's recap the song!";
                feedbackContainer.innerHTML += "<br>";
                feedbackContainer.innerHTML += "Press  <strong><b>SPACE</b></strong> to start";
                startTime = new Date();
                document.getElementById('part1').style.display = 'none'; 
                document.getElementById('phase2transition').textContent = 'Be as accurate as possible.'  //heading
                document.getElementById('part2').style.display = 'block';
                document.querySelector('#experiment .part2instructions').style.display = 'block';
                
                document.querySelector('#experiment .part1instructions').style.display = 'none';
                // console.log('phase trans', phasetransition);
                // console.log('current page', currentPage);
                currentPage = 'experimentMelody';
                // console.log('current page', currentPage);
                document.getElementById('notesfeedback').textContent = `Notes: 0`;
                // expectingUserInput = true;
                // ContinuebuttonResult.disabled = true;
                // console.log("phase transition")
                phasetransition = false
                // Add event listener for space key to trigger the next phase
                document.addEventListener('keydown', handleKeyPress);
                // birdContainer.style.display = 'none';
                pageSwitcher('experimentResult', 'experiment');
                return;
    


            }
            // document.removeEventListener('keydown', handleKeyPress);

            // feedbackContainer.innerHTML = "Let's recap the song!";
            // feedbackContainer.innerHTML += "<br>";
            // feedbackContainer.innerHTML += "Press  <strong><b>SPACE</b></strong> to start";

            // timerContainer.style.display = 'block';
            // buttonContainer.style.display = 'block';

            // startTime = new Date();
            // document.getElementById('part1').style.display = 'none';   //heading
            // document.getElementById('part2').style.display = 'block';
            // document.querySelector('#experiment .part2instructions').style.display = 'block';
            // document.querySelector('#experiment .part1instructions').style.display = 'none';
            // console.log('phase trans', phasetransition);
            // console.log('current page', currentPage);
            // currentPage = 'experimentMelody';
            // console.log('current page', currentPage);


            const continueButton = document.querySelector(' #buttonContainer .btn:nth-child(1)');
            continueButton.disabled = true;
            // expectingUserInput = true;


            document.getElementById('notesfeedback').textContent = `Notes: 0`;
            // expectingUserInput = true;
            // ContinuebuttonResult.disabled = true;
            // console.log("phase transition")

            // Add event listener for space key to trigger the next phase
            document.addEventListener('keydown', handleKeyPress);
            // birdContainer.style.display = 'none';
            pageSwitcher('birdreply', 'experiment');
            return;

        }
        else if (currentPage === 'experimentPhase3') {
            birdContainer.style.display = 'none';
            experimentEnded = false;
            currentPage = 'MelodyResult';
            const currentMelodyBox = melodyBoxes[currentPhase];
            const subMelodies = currentMelodyBox.querySelectorAll('.submelody-box');
            const currentSubmelody = subMelodies[currentMelodyIndex - 1];
            const currentPhasebox = currentSubmelody.querySelectorAll('.phase-box')[2];
            currentPhasebox.classList.remove('active-phase');
            currentPhasebox.classList.add('completed-phase');
            roundScores.push(score);
            birdinstruction = false;
            // console.log("score-1", score);

            // Assuming `data` is the object you have with keys like "3-1", "3-2", ..., "3-4"
            let total = 0;
            let specifiedPrefix = currentPhase.toString();; // Change this to the desired prefix (e.g., "3-" for keys starting with "3")
            // console.log(currentPhase)
            for (let key in RoundPhaseScores) {
                if (key.startsWith(specifiedPrefix)) { // Check if the key starts with the specified prefix
                    let phaseData = RoundPhaseScores[key]; // e.g., { phase1: [93.75], phase2: [31.25], phase3: [41.776315789473685] }

                    // Loop through each phase in phaseData
                    for (let phaseKey in phaseData) {
                        total += phaseData[phaseKey][0]; // Add the value in the array (e.g., 93.75)
                    }
                }
            }

            // console.log("Total:", total);
            let maxScore = 1100;
            let maxBonus = 0.8;
            // Calculate the bonus based on the proportion of the score relative to the max score
            let bonus = (total / maxScore) * maxBonus;
            // console.log("bonus", bonus);
            roundBonus.push(bonus);



            // document.getElementById('melodyPartResult').innerHTML = `${Math.round(bonus)}${fakecurrency[currentPhase]}`;

            document.getElementById('part3').style.display = 'none';
            // console.log("currentMelodyindex", currentMelodyIndex)
            if (!phaseMelodies[currentMelodyIndex + 1]) {
                currentPage = 'MelodyResult';

                // console.log("currentPhase", currentPage);
                feedbackContainer.textContent = ` 🎵What's next`;
                // document.getElementById('buttonMelodyResult').textContent = 'Continue';
                const continueButton = document.querySelector(' #buttonContainer .btn:nth-child(1)');
                continueButton.disabled = true;
                pageSwitcher('experimentResult', 'MelodyResult');
                document.getElementById('MelodyNum').textContent = currentPhase + 1;
                // console.log("totalphaseTime", totalPhaseTime);
                let totalTime = totalPhaseTime.reduce((accumulator, currentValue) => accumulator + currentValue, 0);
                // console.log("", totalTime); // Outputs the sum of all times in the list
                // document.getElementById('MelodyTotalTime').textContent = totalTime;
                MelodyTotalTime.push(totalTime);

                // let melodyScores = roundScores.reduce((accumulator, currentValue) => accumulator + currentValue, 0) / roundScores.length;
                // document.getElementById('melodyTotalResult').innerHTML = `${bonus.toFixed(2)}&#163;`;  
                document.getElementById('melodyTotalResult').innerHTML = `${((bonus / 0.8) * 100).toFixed(2)}%`;
                specifiedPrefix = currentPhase.toString();;
                let percent = 0;
                for (let key in RoundPhaseScores) {
                    if (key.startsWith(specifiedPrefix)) { // Check if the key starts with the specified prefix
                        let phaseData = RoundPhaseScores[key]; // e.g., { phase1: [93.75], phase2: [31.25], phase3: [41.776315789473685] }

                        // Loop through each phase in phaseData
                        for (let phaseKey in phaseData) {
                            percent += phaseData[phaseKey][0]; // Add the value in the array (e.g., 93.75)
                        }
                    }
                }

                // let partPercentage = roundPercentage.slice(-2)
                // let averagePartPercentage = partPercentage.reduce((acc, val) => acc + val, 0) / 2;
                // document.getElementById('PartTotalPercentage').textContent = Math.round(averagePartPercentage);
                // let totalPercentage = percent/11;
                // console.log(percent);
                document.getElementById('totalPercentage').textContent = Math.round(percent / 11);

                const hiddenUl = document.querySelector('ul[style="display: none;"]');
                hiddenUl.style.display = 'block';
                return;
            }

            pageSwitcher('experimentResult', 'MelodyResult');
            // console.log('----')
            ContinuebuttonResult.disabled = false;


            return;
        }
        else if (currentPage === 'MelodyResult') {
            experimentEnded = false;
            currentPage = 'experiment';
            // console.log("Melodypage")
            score = 0;
            document.getElementById('score').style.display = 'block';
            document.getElementById('notesfeedback').textContent = `Notes: 0`;
            // timerContainer.style.display = 'none';
            buttonContainer.style.display = 'none';
            document.getElementById('score-value').textContent = score;


            document.querySelector('#experiment .part3instructions').style.display = 'none';

            document.querySelector('#experiment .part1instructions').style.display = 'block';
            document.getElementById('part1').style.display = 'block';

            // console.log(currentMelodyIndex);
            // console.log(phaseMelodies[currentMelodyIndex]);
            if (!phaseMelodies[currentMelodyIndex + 1]) {
                currentPage = 'MelodyResult';
                currentPhase++;
                // console.log("currentPhase", currentPhase);
                currentMelodyIndex = 0;
                // console.log("currentMelodyIndexMelodypage", currentMelodyIndex);
                totalPhaseTime = [];
                roundPercentage = [];
                roundScores = [];

                if (currentPhase <= 4) {
                    currentPage = 'experiment';
                    phaseMelodies = [];

                    currentMelodyIndex = 0;
                    // console.log("currentMelodyIndex", currentMelodyIndex);
                    // also check currentnoteindex
                    // console.log("phaseMEldoiesnext round", phaseMelodies);
                    feedbackContainer.textContent = `Melody ${currentPhase + 1} Part ${currentMelodyIndex + 1}.<br> Press  <strong><b>SPACE</b></strong>`;
                    startTime = new Date();
                    // console.log("starting new time")
                    totalPhaseTime = [];
                    // console.log('TotalPhasetime', totalPhaseTime)
                    // Change the bird image based on the current phase number
                    const birdImage = document.getElementById('bird-img');
                    birdImage.src = `./src/images/bird${birdPic[currentPhase]}.png`;
                    // console.log("birdImage", birdImage);
                    const birdResImage = document.getElementById('birdResult-img');
                    birdResImage.src = `./src/images/bird${birdPic[currentPhase]}.png`;
                    const MelodyImage = document.getElementById('MelodyBird');
                    MelodyImage.src = `./src/images/bird${birdPic[currentPhase]}.png`;

                    const birdReply = document.getElementById('birdreply-img');
                    birdReply.src = `./src/images/bird${birdPic[currentPhase]}.png`;

                    const phaseImg = document.getElementById('phase-img');
                    phaseImg.src = `./src/images/bird${birdPic[currentPhase]}.png`;

                    // change bird icon
                    // setTimeout(displayNextNote, 1000);   //add this to result page later it is already in start phase funcr
                    // ContinuebuttonResult.disabled = true;
                    const hiddenUl = document.querySelector('ul[style="display: block;"]');

                    if (hiddenUl) {
                        hiddenUl.style.display = 'none';
                    }
                    // console.log('Changing melody !')
                    document.getElementById('MelodyNo').textContent = currentPhase + 1;
                    const melImage = document.getElementById('Melody-img');
                    melImage.src = `./src/images/bird${birdPic[currentPhase]}.png`;
                    pageSwitcher('MelodyResult', 'Melodychange');
                    // pageSwitcher('MelodyResult', 'experiment');
                    return;
                }
                else {
                    currentPage = 'result';
                    // console.log('kdynnkfkjnsn')
                    reward = roundBonus.reduce((accumulator, currentValue) => accumulator + currentValue, 0);
                    // console.log("reward", reward.toFixed(2));
                    
               
                    console.log("reward", reward);
            
                   
                    handleNavigation(workerID);     //to send data                
              
                   
                 
                }
            }
            // console.log('switching')
            pageSwitcher('experimentResult', 'experiment');
            // ContinuebuttonResult.disabled = true;
            return;
        }
        else if (currentPage === 'Instruction4') {
            currentPage = 'Instruction4b';
            score = 0;
            // window.currentTimerInterval = startTimer(60);
            feedbackInstruction.textContent = `Do you remember the song?`;
            timerInstruction.style.display = 'block';
            buttonInstructionContainer.style.display = 'block';
            // add score and percent calculations

            document.getElementById('scoreInstruction').style.display = 'none';
            const continueButton = document.querySelector(' #buttonInstructionContainer .btn:nth-child(1)');
            continueButton.disabled = true;

            expectingUserInput = true;
            ContinuebuttonResult.disabled = true;
            ContinueInstructionbuttonResult.disabled = true;
            pageSwitcher('instructionResult', 'Instruction4');
            const manualBulletDiv = document.querySelector('.manual-bullets');
            manualBulletDiv.style.display = 'none';
            const part1heading = document.getElementById('Part1');
            part1heading.style.display = 'none';
            const part2heading = document.getElementById('Part2');
            part2heading.style.display = 'block';
            // console.log('here')
            return;
        } else if (currentPage === 'Instruction4b') {
            currentPage = 'Instruction4c';
            pageSwitcher('instructionResult', 'Instruction4');
            // window.currentTimerInterval = startTimer(60);
            feedbackInstruction.textContent = `Predict the next melody? You have 10 notes!!!!!`;
            const part2heading = document.getElementById('Part2');
            part2heading.style.display = 'none';
            const part3heading = document.getElementById('Part3');
            part3heading.style.display = 'block';
            expectingUserInput = true;
            ContinueInstructionbuttonResult.disabled = true;
            const continueButton = document.querySelector(' #buttonInstructionContainer .btn:nth-child(1)');
            continueButton.disabled = true;

        }
        else if (currentPage === 'Instruction4c') {
            let PartTimes = totalPhaseTime.slice(-3);
            // Sum the last 3 elements
            let sumOfPartTimes = PartTimes.reduce((accumulator, currentValue) => accumulator + currentValue, 0);
            let percent = (score / 100) * 100;
            // console.log("percent phase 3", percent)
            roundPercentage.push(percent);
            document.getElementById('InstructionTimes').textContent = sumOfPartTimes;
            // console.log("Part times", sumOfPartTimes);

            let partPercentage = roundPercentage.slice(-3)
            let averagePartPercentage = partPercentage.reduce((acc, val) => acc + val, 0) / 3;
            document.getElementById('InstructionTotalPercentage').textContent = Math.round(averagePartPercentage);
            document.getElementById('buttonMelodyResult').textContent = 'Go to comprehension questions';
            // console.log("roundscores")
            let partScores = roundScores.slice(-3)
            let bonus = partScores.reduce((accumulator, val) => accumulator + val, 0) / 3;
            document.getElementById('InstructionmelodyPartResult').innerHTML = `${Math.round(bonus)}${fakecurrency[currentPhase - 1]}`;

            pageSwitcher('instructionResult', 'InstructionMelodyResult');
            ContinueInstructionbuttonResult.disabled = true;
        }
        else if (currentPage === 'InstructionMelodyResult') {
            pageSwitcher('InstructionMelodyResult', 'Instruction11');
            return;

        }
    }

    // Function to remove all incorrect notes
    function removeAllIncorrectNotes() {
        const incorrectNotes = document.querySelectorAll('.note.incorrect'); // Select all elements with 'note incorrect' class
        incorrectNotes.forEach(note => note.remove()); // Remove each incorrect note
        // console.log('All incorrect notes removed.');
    }


    function recordUserInput(input, noteElement) {
        let expectedNote;
        if (currentPage != 'Instruction8start' && currentPage != 'Instruction9start' && currentPage != 'Instruction10start') {
            expectedNote = numToLetter[phaseMelodies[currentMelodyIndex][currentNoteIndex]];
            // console.log("expectedNote", expectedNote);
        }

        if (insTransition) {
            return;
        }


        if (currentPage != 'experiment' && currentPage != 'Instruction8start') {
            // console.log("currentPage", currentPage)
            // console.log("increasing index", currentNoteIndex);
            currentNoteIndex++;
        }

        if (currentPage != 'Instruction8start' && currentPage != 'Instruction9start' && currentPage != 'Instruction10start') {
            // console.log('currentPAge', currentPage)
            document.getElementById('notesfeedback').textContent = `Notes: ${currentNoteIndex}/${phaseMelodies[currentMelodyIndex].length}`;
        }




        if (currentPage === 'Instruction9start') {
            // console.log("changing note no")
            document.getElementById('notesInsfeedback').textContent = `Notes: ${currentNoteIndex}/${sample_Melody[currentMelodyIndex].length}`;
        } else if (currentPage === 'Instruction10start') {
            document.getElementById('notesInsfeedback').textContent = `Notes: ${currentNoteIndex}/${sample_Melody[currentMelodyIndex].length}`;
        }


        // console.log("currentNote", currentNoteIndex)
        // console.log('currentpage', currentPage)

        if (currentPage === 'experimentPhase3') {
            // console.log("pointsPrenote",pointsPerNote)
            startTime = new Date();      //correct date
            feedbackContainer.textContent = 'What will come next?';
            const line = document.querySelector(`.line[data-line="${lineData}"]`);
            noteElement = document.createElement('div');
            noteElement.className = 'note correct';
            noteElement.style.top = '0%';
            noteElement.style.left = line.style.left;
            pointsPerNote = 100 / (phaseMelodies[currentMelodyIndex].length);
            expectingUserInput = true
            if (input === expectedNote) {
                if (score + pointsPerNote > 100) {
                    score = 100;
                } else {
                    score += pointsPerNote;
                }
            }
        }
        else if (currentPage === 'Instruction10start') {
            startTime = new Date();      //correct date
            instructionFeedback.textContent = 'What will come next?';
            const line = document.querySelector(`.line[data-line="${lineData}"]`);
            noteElement = document.createElement('div');
            noteElement.className = 'note correct';
            noteElement.style.top = '0%';
            noteElement.style.left = line.style.left;
            pointsPerNote = Math.round(100 / (sample_Melody[currentMelodyIndex].length));
            expectingUserInput = true
            if (input === expectedNote) {
                if (score + pointsPerNote > 100) {
                    score = 100;
                } else {
                    score += pointsPerNote;
                }
            }
          
                if (currentNoteIndex >= sample_Melody[currentMelodyIndex].length) {   //check this again
                    expectingUserInput = false;
                    // console.log('ending exper')
                    currentPage = 'Instruction11';
                    setTimeout(() => {
                    // birdContainer.style.display = 'none';
                    document.getElementById('instructionDemo').style.display = 'none';
                    pageSwitcher('Instruction10', 'Instruction11');
                }, 1000);

                    return;
                }
         


        }



        else if (currentPage === 'experimentPhase2') {
            feedbackContainer.textContent = 'Play back the melody from memory!';
            const line = document.querySelector(`.line[data-line="${lineData}"]`);
            noteElement = document.createElement('div');
            noteElement.className = 'note correct';
            noteElement.style.top = '0%';
            noteElement.style.left = line.style.left;
            startTime = new Date();      //correct date
            pointsPerNote = 100 / (phaseMelodies[currentMelodyIndex].length);
            // console.log("pointsPernote", pointsPerNote);
            expectingUserInput = true
            if (input === expectedNote) {
                if (score + pointsPerNote > 100) {
                    score = 100;
                } else {
                    score += pointsPerNote;
                }
            }
        }
        else if (currentPage === 'Instruction9start') {
            instructionFeedback.textContent = 'Do you remember the song?';
            const line = document.querySelector(`.line[data-line="${lineData}"]`);
            // console.log('line', line);
            noteElement = document.createElement('div');
            noteElement.className = 'note correct';
            noteElement.style.top = '0%';
            noteElement.style.left = line.style.left;
            startTime = new Date();      //correct date
            pointsPerNote = Math.round(100 / (sample_Melody[currentMelodyIndex].length));
            // console.log("pointsPernote", pointsPerNote);
            expectingUserInput = true
            if (input === expectedNote) {
                if (score + pointsPerNote > 100) {
                    score = 100;
                } else {
                    score += pointsPerNote;
                }
            }
            setTimeout(() => {
                if (currentNoteIndex >= sample_Melody[currentMelodyIndex].length) {   //check this again
                    // console.log('ending exper')
                    // birdContainer.style.display = 'none';
                    document.getElementById('instructionDemo').style.display = 'none';
                    pageSwitcher('Instruction9', 'Instruction10');

                    return;
                }
            }, 1000);


        }
        else if (currentPage === 'Instruction8start') {
            expectedNote = numToLetter[sample_Melody[currentMelodyIndex][currentNoteIndex]];
            // console.log("expectedNote", expectedNote);
            pointsPerNote = Math.round(100 / (sample_Melody[currentMelodyIndex].length));
            if (input === expectedNote) {
                expectingUserInput = false;
                spaceActive = true;
                const existingNotes = document.querySelectorAll('.note');

                // console.log("existing notes", existingNotes);
                const lineHeight = line.offsetHeight;
                // console.log("increasing index", currentNoteIndex);
                currentNoteIndex++;
                // console.log("currentNoteIndex", currentNoteIndex);

                expectingUserInput = false;
                noteElement.className = 'note correct';

                if (score + pointsPerNote > 100) {
                    score = 100;
                } else {
                    score += pointsPerNote;
                }
                expectingUserInput = false;
                spaceActive = true;
                removeAllIncorrectNotes()
                document.getElementById('instructionFeedback').innerHTML = `<strong style="color: darkgreen;">Correct!</strong>`;
                // document.getElementById('instructionDemo').style.display = 'none';
                birdFeedback.innerHTML = `<strong style="color: darkgreen;">Correct!</strong>`;
                setTimeout(displayInsNote, 500);

                // birdContainer.style.display = 'block';
                if (currentNoteIndex >= sample_Melody[currentMelodyIndex].length) {
                    spaceActive = false;
                }

                setTimeout(() => {
                    if (currentNoteIndex >= sample_Melody[currentMelodyIndex].length) {   //check this again
                        // console.log('ending exper')
                        // birdContainer.style.display = 'none';
                        document.getElementById('instructionDemo').style.display = 'none';
                        pageSwitcher('Instruction8', 'Instruction9');

                        return;
                    }
                }, 1000);


            } else {
                removeKeyListeners();
                // expectingUserInput = false;
                // noteElement.remove();
                noteElement.className = 'note incorrect';

                expectingUserInput = true;
                wrongNote = true;
                // spaceActive = true;

                noteElement.className = 'note incorrect';


                document.getElementById('instructionFeedback').innerHTML = `<strong style="color: darkred;">Wrong!</strong>`;

                expectingUserInput = true;


                setTimeout(() => {
                    if (currentNoteIndex >= sample_Melody[currentMelodyIndex].length) {   //check this again see if we need this??
                        // console.log('ending exper')
                        spaceActive = false;

                        pageSwitcher('Instruction8start', 'Instruction9');
                        return;
                    }

                    // expectingUserInput = true;
                }, 1000);

                score -= pointsPerNote;
                birdFeedback.innerHTML = `<strong style="color: darkred;">Wrong!</strong>`;
                // console.log('wrongnoteScore', wrongnoteScore);
                // console.log('pointsPerNote', pointsPerNote); //reduce score
                document.getElementById('score-value').textContent = score;
                document.getElementById('score-value').style.color = 'darkred';

                addKeyListeners();

                // console.log('Note to Line Mapping:', noteToLineMap);
                return;

            }


        }

        else if (currentPage === 'experiment') {
            pointsPerNote = 100 / (phaseMelodies[currentMelodyIndex].length);
            // console.log("experimentrecord")
            // // wrongnoteScore = Math.round(pointsPerNote / 6);
            // console.log('wrongnoteScore', wrongnoteScore);
            // console.log('pointsPerNote', pointsPerNote);

            if (input === expectedNote) {
                // console.log('expectedNote', expectedNote)

                expectingUserInput = false;
                spaceActive = true;


                // console.log("existing notes", existingNotes);
                const lineHeight = line.offsetHeight;

                currentNoteIndex++;
                // console.log("currentNoteIndex", currentNoteIndex);

                expectingUserInput = false;


                if (score + pointsPerNote > 100) {
                    score = 100;
                } else {
                    // console.log('pointsperNote', pointsPerNote);
                    score += pointsPerNote;
                }
                removeAllIncorrectNotes()

                document.getElementById('score-value').textContent = Math.round(score);
                noteElement.className = 'note correct';
                feedbackContainer.innerHTML = `<strong style="color: darkgreen;">Correct!</strong><br> `;
                // birdFeedback.innerHTML = `<strong style="color: darkgreen;">Correct!</strong> Press <strong>Space</strong>  to continue`;   //add unstruczionnote logic
                document.getElementById('score-value').style.color = 'darkgreen';


                setTimeout(displayNextNote, 500);

                setTimeout(() => {
                    startTime = new Date();
                    expectingUserInput = true;

                }, 500);


                // spaceActive = true;
                // experiment.style.display = 'none';
                document.getElementById('notesfeedback').textContent = `Notes: ${currentNoteIndex}/${phaseMelodies[currentMelodyIndex].length}`;
                // birdContainer.style.display = 'block';


                if (currentNoteIndex >= phaseMelodies[currentMelodyIndex].length) {
                    spaceActive = false;
                }

                setTimeout(() => {
                    if (currentNoteIndex >= phaseMelodies[currentMelodyIndex].length) {   //check this again
                        // console.log('ending exper')
                        // birdContainer.style.display = 'none';

                        endExperiment();

                        return;
                    }


                }, 1000);


            } else {

                removeKeyListeners();
                // expectingUserInput = false;
                // noteElement.remove();
                noteElement.className = 'note incorrect';
                // feedbackContainer.textContent = `Wrong! The correct note was ${expectedNote}`;   
                feedbackContainer.innerHTML = `<strong style="color: darkred;">Wrong!</strong>`;    //add unstruczionnote logic
                startTime = new Date();
                expectingUserInput = true;
                // console.log('expectingUserInput', expectingUserInput);
                // const existingNotes = document.querySelectorAll('.note');
                // if (existingNotes.length > 0) {
                //     const lastNote = input;
                //     // lastNote.style.backgroundColor = 'red';

                // }
                // wrongNote = true;
                // spaceActive = true;
                // experiment.style.display = 'none';
                // birdContainer.style.display = 'block';

                setTimeout(() => {
                    if (currentNoteIndex >= phaseMelodies[currentMelodyIndex].length) {   //check this again see if we need this??
                        console.log('ending exper')
                        spaceActive = false;

                        endExperiment();
                        return;
                    }

                    // expectingUserInput = true;
                }, 1000);

                score -= pointsPerNote;
                // console.log('wrongnoteScore', wrongnoteScore);
                // console.log('pointsPerNote', pointsPerNote); //reduce score
                document.getElementById('score-value').textContent = Math.round(score);
                document.getElementById('score-value').style.color = 'darkred';

                addKeyListeners();



                // console.log('Note to Line Mapping:', noteToLineMap);
                return;


            }

        }
    }
    function endExperiment() {
        // console.log("called end experiment");
        endPhaseTime = new Date();  // Correct variable name with uppercase 'T'

        // Calculate total time in milliseconds
        const totalTime = endPhaseTime - startPhaseTime;
        totalPhaseTime.push(totalTime);

        // Adjust the key for storing data
        let currentKey;
        if (currentPage === 'experiment' || currentPage === 'experimentPhase2') {
            phasetransition = true
            document.getElementById('notesfeedback').textContent = `Notes: ${currentNoteIndex}/${phaseMelodies[currentMelodyIndex].length}`;
            // console.log("phasetransition")
        }



        if (currentPage === 'experimentPhase3') {
            // Use currentMelodyIndex - 1 for phase 3
            currentKey = `${currentPhase}-${currentMelodyIndex - 1}`;
        } else {

            // Use currentMelodyIndex for phases 1 and 2
            currentKey = `${currentPhase}-${currentMelodyIndex}`;
        }

        // Initialize the object for this key if it doesn't exist
        if (!RoundTimePhase[currentKey]) {
            RoundTimePhase[currentKey] = {};
            // console.log('Initialized RoundTimePhase for', currentKey);
        }
        if (!RoundPhaseScores[currentKey]) {
            RoundPhaseScores[currentKey] = {};
            // console.log('Initialized RoundPhaseScores for', currentKey);
        }

        if (!roundKeyPhase[currentKey]) {
            roundKeyPhase[currentKey] = {};
            // console.log('Initialized roundKeyPhase for', currentKey);
        }


        // Store data according to the current phase
        if (currentPage === 'experiment') {
            roundScores.push(score);
            // console.log('Storing data for phase 1');
            RoundPhaseScores[currentKey]['phase1'] = roundScores.slice()
            RoundTimePhase[currentKey]['phase1'] = RoundtimePresses.slice();
            // console.log('RoundTimePhase phase1', RoundTimePhase);
            roundKeyPhase[currentKey]['phase1'] = roundPressedKeys.slice();
            // console.log('roundKeyPhase phase1', roundKeyPhase);
            document.getElementById('notesfeedback').textContent = `Notes: ${currentNoteIndex}/${phaseMelodies[currentMelodyIndex].length}`;
        } else if (currentPage === 'experimentPhase2') {
            roundScores.push(score);
            // console.log('Storing data for phase 2');
            RoundPhaseScores[currentKey]['phase2'] = roundScores.slice()
            RoundTimePhase[currentKey]['phase2'] = RoundtimePresses.slice();
            roundKeyPhase[currentKey]['phase2'] = roundPressedKeys.slice();
        } else if (currentPage === 'experimentPhase3') {
            roundScores.push(score);
            // console.log('Storing data for phase 3');
            RoundPhaseScores[currentKey]['phase3'] = roundScores.slice()
            RoundTimePhase[currentKey]['phase3'] = RoundtimePresses.slice();
            roundKeyPhase[currentKey]['phase3'] = roundPressedKeys.slice();
            // Download the combined data as a single JSON file
            // downloadJSON(combinedData, "round_data.json");

        }

        // console.log('RoundTimePhase', RoundTimePhase);
        // console.log('roundKeyPhase', roundKeyPhase);
        // console.log('RoundPhaseScores', RoundPhaseScores);

        // console.log("roundPressedkeys", roundPressedKeys);
        // console.log("roundtimrkeys", RoundtimePresses);
        // console.log("RoundPhaseScores", RoundPhaseScores);

        // Reset RoundtimePresses and roundPressedKeys for the next phase
        RoundtimePresses = [];
        roundPressedKeys = [];
        roundScores = [];

        // console.log("total time", totalPhaseTime);

        // for instruction page
        // console.log('current page', currentPage);
        if (currentPage === 'experiment') {
            document.getElementById('scoreFeedback').innerHTML = `You scored ${Math.round(score)}/100 points! <br> Press <strong><b>SPACE</b></strong> to phase2!`;
        }
        else if (currentPage === 'experimentPhase2' && currentMelodyIndex < 4) {
            document.getElementById('scoreFeedback').innerHTML = `You scored ${Math.round(score)}/100 points! <br> Press <strong><b>SPACE</b></strong> to next part`;
        }
        else if (currentPage === 'experimentPhase2' && currentMelodyIndex == 4) {
            document.getElementById('scoreFeedback').innerHTML = `You scored ${Math.round(score)}/100 points! <br> Press <strong><b>SPACE</b></strong> to phase3!`;
        }
        else if (currentPage === 'experimentPhase3') {
            document.getElementById('scoreFeedback').innerHTML = `You scored ${Math.round(score)}/100 points! <br> Press <strong><b>SPACE</b></strong> to see your overall score!`;
        }
        pageSwitcher('experiment', 'experimentResult');


        expectingUserInput = false;
        clearNotes();

        // Remove all existing timers
        // clearTimer(); // Clear all timers at the end of the experiment


    }


    document.addEventListener('keydown', handleKeyPress);
    const experimentSection = document.getElementById('experiment');
    const keys = experimentSection.querySelectorAll('.key');


    // A function to remove all event listeners
    function removeKeyListeners() {
        keys.forEach(key => {
            key.removeEventListener('click', handleKeyPress);
            document.removeEventListener('keydown', handleKeyPress);
            // console.log('removed')
        });
    }

    // A function to add all event listeners only once per click
    function addKeyListeners() {

        keys.forEach(key => {
            key.addEventListener('click', handleButtonClick);
            // Attach the keydown event listener
            document.addEventListener('keydown', handleKeyPress);
            // console.log('added');
        });


    }



    addKeyListeners();  // Initial adding of event listenersent listeners





    function setupContinueButton() {
        const continueButton = document.querySelector('#buttonContainer .btn:nth-child(1)');



        // Add click event listener to the continue button
        continueButton.addEventListener('click', handleContinueClick);

        // Add keydown event listener to the document to listen for space key
        document.addEventListener('keydown', (event) => {
            event.preventDefault()
            event.key = ''
            if (event.code === 'Space' || event.code === ' ') {
                event.key = '';
                if (continueButton && continueButton.offsetWidth > 0 && continueButton.offsetHeight > 0 && !continueButton.disabled) {
                    // Check if the button is visible and not disabled before handling the event
                    handleContinueClick();
                }
            }
        });

        function handleContinueClick() {
            // Log the event
            // console.log('Continue button clicked');

            // End the experiment or proceed to the next step
            endExperiment();
        }
    }

    setupContinueButton();



    function setupInstructionContinueButton() {
        const continueButton = document.querySelector(' #InstructionbuttonContainer .btn:nth-child(1)');

        continueButton.addEventListener('click', handleContinueClick);
        document.addEventListener('keydown', (event) => {
            event.preventDefault()
            if (event.code === 'Space' || event.code === ' ') {
                if (continueButton && continueButton.offsetWidth > 0 && continueButton.offsetHeight > 0 && !continueButton.disabled) {
                    // Check if the button is visible and not disabled before handling the event
                    handleContinueClick();
                }
            }
        });
        function handleContinueClick() {
            // Remove the event listener immediately after it's triggered


            // console.log('Continue button clicked');
            if (currentPage === 'Instruction9start') {
  
                pageSwitcher('instructionDemo', 'Instruction10');
                return;
            } else if (currentPage === 'Instruction10start') {
            
                pageSwitcher('instructionDemo', 'Instruction11');
                return;
            }

        }
    }
    setupInstructionContinueButton()



    const combinedData = {
        roundKeyPhase: roundKeyPhase,
        RoundTimePhase: RoundTimePhase,
        RoundPhaseScores: RoundPhaseScores,
        sampled_indices: sampled_indices,
        shift: roundShifts
    };


    // Function to download data as JSON
    function downloadJSON(data, filename) {
        const jsonData = JSON.stringify(data, null, 2); // Convert to JSON with pretty-printing
        const blob = new Blob([jsonData], { type: "application/json" });
        const url = URL.createObjectURL(blob);

        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        a.click();

        URL.revokeObjectURL(url); // Clean up URL object
    }

    async function handleNavigation(workerID) {
        try {
            await sendData(); // Wait for sendData to complete
            window.location.href = `survey.html?workerID=${workerID}`; // Execute this after sendData finishes
        } catch (error) {
            console.error("Error in sendData:", error);
        }
    }

    function sendData() {
        return new Promise((resolve, reject) => {
            const experimentData = {
                roundKeyPhase: roundKeyPhase,
                RoundTimePhase: RoundTimePhase,
                RoundPhaseScores: RoundPhaseScores,
                sampled_indices: sampled_indices,
                shift: roundShifts,
            };
    
            const formattedReward = reward.toFixed(2);

            if (scenarioId === 0 || window.location.hostname.endsWith("github.io")) {
                console.info("Preview mode: skipping data save.");
                resolve("preview");
                return;
            }
    
            // Create FormData
            const formdata = new FormData();
            formdata.append("action", "completeScenario");
            formdata.append("workerID", workerID);
            formdata.append("assignmentID", assignmentID);
            formdata.append("experimentData", JSON.stringify(experimentData));
            formdata.append("reward", formattedReward);
            formdata.append("scenarioId", scenarioId);
    
            // Create XMLHttpRequest
            const xhr = new XMLHttpRequest();
            xhr.open("POST", "databasecall.php", true);
    
            // Set up response handler
            xhr.onreadystatechange = function () {
                if (xhr.readyState === 4) { // Request is done
                    if (xhr.status === 200) { // HTTP OK
                        console.log("Server response:", xhr.responseText);
                        resolve(xhr.responseText); // Resolve the Promise
                    } else {
                        console.error(`XHR error: Status ${xhr.status}, Response: ${xhr.responseText}`);
                        reject(new Error(`XHR error: Status ${xhr.status}`)); // Reject the Promise
                    }
                }
            };
    
            // Send the request
            xhr.send(formdata);
        });
    }






    // function sendData() {
    //     const experimentData = {
    //         roundKeyPhase: roundKeyPhase,
    //         RoundTimePhase: RoundTimePhase,
    //         RoundPhaseScores: RoundPhaseScores,
    //         sampled_indices: sampled_indices,
    //         shift: roundShifts,
    //     };
    
    //     const formattedReward = reward.toFixed(2);
    
    //     // Create FormData
    //     const formdata = new FormData();
    //     formdata.append("action", "completeScenario");
    //     formdata.append("workerID", workerID);
    //     formdata.append("assignmentID", assignmentID);
    //     formdata.append("experimentData", JSON.stringify(experimentData));
    //     formdata.append("reward", formattedReward);
    //     formdata.append("scenarioId", scenarioId);
    
    //     // Create XMLHttpRequest
    //     const xhr = new XMLHttpRequest();
    //     xhr.open("POST", "databasecall.php", true);
    
    //     // Set up response handler
    //     xhr.onreadystatechange = function () {
    //         if (xhr.readyState === 4) { // Request is done
    //             if (xhr.status === 200) { // HTTP OK
    //                 console.log("Server response:", xhr.responseText);
    //             } else {
    //                 console.error(`XHR error: Status ${xhr.status}, Response: ${xhr.responseText}`);
    //                 return;
    //             }
    //         }
    //     };
    
    //     // Send the request
    //     xhr.send(formdata);
    // }


    // function sendData() {
    //     const experimentData = {
    //         roundKeyPhase: roundKeyPhase,
    //         RoundTimePhase: RoundTimePhase,
    //         RoundPhaseScores: RoundPhaseScores,
    //         sampled_indices: sampled_indices,
    //         shift: roundShifts,
    //     };
    //  //Initiate AJAX request
    // var ajaxRequest = new XMLHttpRequest();
    // try{
    //     // Opera 8.0+, Firefox, Safari
    //     ajaxRequest = new XMLHttpRequest();
    // } catch (e){
    //     // Internet Explorer Browsers
    //     try{
    //         ajaxRequest = new ActiveXObject("Msxml2.XMLHTTP");
    //     } catch (e) {
    //         try{
    //             ajaxRequest = new ActiveXObject("Microsoft.XMLHTTP");
    //         } catch (e){
    //             // Something went wrong
    //             alert("Your browser broke!");
    //             return false;
    //         }
    //     }}
    //     const formattedReward = reward.toFixed(2);
    
    //     const formdata = new FormData();
    //     formdata.append("action", "completeScenario");
    //     formdata.append("workerID", workerID);
    //     formdata.append("assignmentID", assignmentID);
    //     formdata.append("experimentData", JSON.stringify(experimentData));
    //     formdata.append("reward", formattedReward);
    //     formdata.append("scenarioId", scenarioId);
    
    //     var requestOptions = {
    //         method: 'POST',
    //         body: formdata,
    //         redirect: 'follow'
    //       };
    
    //     fetch("./databasecall.php", requestOptions)
    //     .then(response => response.text())
    //     .then(result => console.log(result))
    //     .catch(error => console.log('error', error));
  
    // }



    // });

    // function senddata() {
    //     // Prepare experiment data
    //     const experimentData = {
    //         roundKeyPhase: roundKeyPhase,
    //         RoundTimePhase: RoundTimePhase,
    //         RoundPhaseScores: RoundPhaseScores,
    //         sampled_indices: sampled_indices,
    //         shift: roundShifts,
    //     };
    
    //     // Ensure reward is properly rounded to 2 decimal places
    //     const formattedReward = reward.toFixed(2);
    
    //     // Log data for debugging purposes
    //     console.log("Preparing to send data:", {
    //         action: "completeScenario",
    //         workerID,
    //         assignmentID,
    //         experimentData,
    //         reward: formattedReward,
    //         scenarioId,
    //     });
    
    //     // Use FormData for submission
    //     const formdata = new FormData();
    //     formdata.append("action", "completeScenario");
    //     formdata.append("workerID", workerID);
    //     formdata.append("assignmentID", assignmentID);
    //     formdata.append("experimentData", JSON.stringify(experimentData)); // Serialize experiment data
    //     formdata.append("reward", formattedReward);
    //     formdata.append("scenarioId", scenarioId);
    
    //     // Request options for the fetch call
    //     const requestOptions = {
    //         method: "POST",
    //         body: formdata,
    //         redirect: "follow", // Handle redirects
    //     };
    
    //     // Perform fetch request
    //     fetch("./databasecall.php", requestOptions)
    //         .then((response) => {
    //             // Check if response is OK
    //             if (!response.ok) {
    //                 throw new Error(`HTTP error! status: ${response.status}`);
    //             }
    //             return response.text(); // Parse response text
    //         })
    //         .then((result) => {
    //             console.log("Server response:", result);
    //             if (result.error) {
    //                 console.error("Error from server:", result.error);
    //             } else {
    //                 console.log("Data successfully sent!");
    //             }
    //         })
    //         .catch((error) => {
    //             console.error("Fetch error occurred:", error);
    //         });
    
    //     // Update UI if needed
    //     console.log("Reward displayed to the user:", formattedReward);
    //     // Example: Update reward display
    //     // document.getElementById('reward').textContent = formattedReward;
    // }




    // function senddata() {
    //     const experimentData = {
    //         'roundKeyPhase': roundKeyPhase,
    //         'RoundTimePhase': RoundTimePhase,
    //         'RoundPhaseScores': RoundPhaseScores,
    //         'sampled_indices': sampled_indices,
    //         // 'surveyData': surveyData,
    //         'shift': roundShifts
    //     };

    //     const formdata = new FormData();
    //     formdata.append("action", "completeScenario");
    //     formdata.append("workerID", workerID);
    //     formdata.append("assignmentID", assignmentID);
    //     formdata.append("experimentData", JSON.stringify(experimentData));
    //     formdata.append("reward", reward.toFixed(2));
    //     console.log("sucess")
    //     formdata.append("scenarioId", scenarioId);

    //     const requestOptions = {
    //         method: 'POST',
    //         body: formdata,
    //         redirect: 'follow'
    //     };

    //     fetch("./databasecall.php", requestOptions)
    //         .then(response => response.text())
    //         .then(result => console.log(result))
    //         .catch(error => console.log('error', error));


    //     // document.getElementById('reward').textContent = roundBonus[Math.floor(Math.random() * roundBonus.length)];

    // }



