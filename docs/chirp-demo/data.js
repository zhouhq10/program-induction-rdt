let reward;
let experimentData;
let assignmentID;
let scenarioId
const PREVIEW_MODE = window.location.hostname.endsWith("github.io") || window.location.protocol === "file:";

var fullurl = document.location.href, //url of incoming MTurk/prolific worker, used to extract workerid and assignmentid
    workerID



// console.log("fullurl",fullurl);

// console.log("workerID",workerID);

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

    function getExperimentData(workerID) {
        if (PREVIEW_MODE) {
            experimentData = {};
            reward = 0;
            assignmentID = "";
            scenarioId = 0;
            return experimentData;
        }
        var ajaxRequest = new XMLHttpRequest();
        ajaxRequest.onreadystatechange = function() {
            if (ajaxRequest.readyState === 4) {
                if (ajaxRequest.status === 200) {
                    // Log raw response for debugging
                    // console.log("Raw response:", ajaxRequest.responseText);
                    // let jsonString = ajaxRequest.responseText.replace(/^\d+/, ''); // Remove leading digits
                    let jsonString = ajaxRequest.responseText.trim(); // Simply trim whitespace, no regex2
                    // console.log("ajaxRequest.responseText",ajaxRequest.responseText)
                    // console.log("jsonString",jsonString)
                    let response = JSON.parse(jsonString);
                    // console.log("response",response)
       
                    try {
                        // Parse and store the experiment data
                        experimentData = JSON.parse(response.experimentData);
                        // console.log(experimentData);
                        reward = response.reward;
                        // console.log("reward",reward);
                        assignmentID = response.assignmentID;
                        scenarioId = response.scenarioId;
                    } catch (error) {
                        console.error("Error parsing JSON:", error);
                    }
                } else {
                    console.error("Error with AJAX request:", ajaxRequest.statusText);
                }
            } };
            var queryString = "?action=getExperimentData&workerID=" + encodeURIComponent(workerID);
            ajaxRequest.open("GET", "databasecall.php" + queryString, true);
            ajaxRequest.send();
        }

//Retrieve worker and assignment id from URL header, and then assigns them a scenario
function beginExperiment() { //called when participant grants consent
    // Retrieve assignmentID, workerID, ScenarioID, and environment from URL
 
    workerID = turkGetParam('workerID');
    // console.log("workerID",workerID)



    // document.getElementById('PROLIFIC_PID').value = workerID; //prepopulate MTurk
    experimentData = getExperimentData(workerID);
  

}

beginExperiment()


// setTimeout(function() {
//     console.log('Experiment data:', experimentData);
//     console.log('Reward:', reward);

//     console.log(workerID);
//     console.log(scenarioId);
//     console.log(assignmentID);
// }, 100);
  
  document.getElementById('SurveySubmit').addEventListener('click', () => {
    // Collect survey data
    const prolificID = document.getElementById('prolific-id').value;
    const age = document.getElementById('age').value;
    const gender = document.querySelector('input[name="gender"]:checked')?.value;
    const strategies = document.getElementById('strategies').value;

    // Validate data (ensure all fields are filled)
    if (!prolificID || !age || !gender || !strategies) {
        alert('Please complete all fields before submitting.');
        return;
    }

    // Save survey data
    const surveyData = {
        prolificID,
        age,
        gender,
        strategies,
    };

    // console.log('Survey Data:', surveyData); // Replace with your API call if needed

    // Add survey data to the extra data for submission
    senddata(surveyData);

    // Navigate to the bonus page
   
    document.getElementById('reward').textContent = Math.round(reward * 100) / 100;
    document.getElementById('rewardTotal').textContent = (Math.round(reward * 100) / 100)+4;
    document.getElementById('survey').style.display = 'none';
    document.getElementById('Bonus').style.display = 'block';
});

function senddata(surveyData) {
    if (PREVIEW_MODE || scenarioId === 0) {
        experimentData = Object.assign(experimentData || {}, { surveyData });
        return;
    }

    // Combine experimentData with surveyData
 
    const extraData = { surveyData };
    Object.assign(experimentData, extraData);

    // console.log('Combined Data:', experimentData);

    // Create FormData for submission
    const formdata = new FormData();
    formdata.append("action", "completeScenario");
    formdata.append("workerID", workerID);
    formdata.append("assignmentID", assignmentID);
    formdata.append("experimentData", JSON.stringify(experimentData));
    formdata.append("reward", reward);
    formdata.append("scenarioId", scenarioId);

    const requestOptions = {
        method: 'POST',
        body: formdata,
        redirect: 'follow',
    };

    // Send the data via a POST request
    fetch("./databasecall.php", requestOptions)
        .then(response => response.text())
        .then(result => console.log(result))
        .catch(error => console.log('error', error));

    // Display the reward and proceed to the final page
   
}
