function loadPlayer(videoUrl, srtUrl) {
    let pConfig = {
        hotkeys: true,
        playlist: false,
        search: false,
        theme: 'vg',
        plugins: ['filmstrip']
    };
    let player = new VG.Player(pCont, pConfig);

    return new Promise((resolve, reject) => {
        player.loadUrl(videoUrl, (err) => {
            // VG.Captions.parseSubs(player.getTimeline(), srtUrl, 'srt', (err, subs) => {
            //     if (err) {
            //         console.error("error parsing subs", err);
            //         return;
            //     }
            //     console.log("SRT OK");
            //     player.addCaptions(subs);
            //     player.play();
            // });
            if (err) {
                reject(err);
            } else {
                resolve(player);
            }
        });
    });

}


function fetchJSONL(jsonUrl) {
    return new Promise((resolve, reject) => {
        let xhr = new XMLHttpRequest();
        xhr.open("GET", jsonUrl, true);
        xhr.onreadystatechange = () => {
            if (xhr.readyState === 4 && xhr.status === 200) {
                let result = xhr.responseText.split("\n")
                    .filter(value => value !== "")
                    .map(value => {
                        return JSON.parse(value);
                    });
                resolve(result);
            } else if (xhr.status >= 400) {
                reject(xhr);
            }
        };
        xhr.send();
    });
}

function argMax(array) {
    return array.map((x, i) => [x, i])
        .reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

function updateLabel(clsBox, labels) {
    var clsId = clsBox.getAttribute("clsId");
    var fn = clsBox.getAttribute("fn");
    var acc = clsBox.getAttribute("acc");
    clsBox.innerText = fn + ": " + labels[clsId] + " " + Math.round(acc * 100) / 100;
}

let url = new URL(document.URL);
let id = url.searchParams.get("id");
let videoURL = "/vg_smoke/" + id;
let jsonlUrl = "/vg_smoke/jsonl/" + id + "/result.jsonl";

function drawChart(predByFrame) {
    var posData = [];
    var negData = [];
    var thresholdData = [];
    var xLabels = [];
    predByFrame.forEach(item => {
        xLabels.push(item[0]);
        var no = item[1][0];
        var yes = item[1][1];

        // if (yes < 0.8) {
        //     yes = 0;
        // }
        // if (no < 0.8) {
        //     no = 0;
        // }

        negData.push(no);
        posData.push(yes);
        thresholdData.push(0.8)
    });

    var ctx = document.getElementById('chart1').getContext('2d');
    var myChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: xLabels,
            datasets: [
                {
                    label: "Threshold",
                    data: thresholdData,
                    borderColor: "#fff314",
                    // backgroundColor: "#ff0000"
                }, {
                    label: "No Smoking",
                    data: negData,
                    borderColor: "#3fae06",
                    // backgroundColor: "#ff0000"
                },
                {
                    label: "Smoking",
                    data: posData,
                    borderColor: "#ff1597",
                    // backgroundColor: "#00FF00"
                },
            ]
        },
        options: {
            responsive: true,
            // title: {
            //     display: true,
            //     text: 'Chart.js Line Chart - Logarithmic'
            // },
            tooltips: {
                enabled: true,
                // callbacks: {
                //     label: (tooltipItem, data) => {
                //         let fn = tooltipItem.xLabel;
                //         console.log(tooltipItem, fn);
                //         player.seekFrame(fn);
                //     }
                // }
            }
        }
    });
}

loadPlayer(videoURL).then(player => {
    window.player = player;
    let totalClasses = 2;
    let labels = ["NO SMOKING", "SMOKING"];
    window.yByFrame = [];
    fetchJSONL(jsonlUrl).then(yByFrame => {
        console.log(jsonlUrl);
        window.yByFrame = yByFrame;
        drawChart(yByFrame);
    }).catch(reason => {
        console.error(reason);
    });

    // let timebox = document.querySelector(".vg_time_controls_box");
    // clsBox.setAttribute("class", "vg_aligner vg_info");
    // timebox.appendChild(clsBox);

    var clsBox = document.getElementById("clsLabel");
    clsBox.addEventListener("click", ev => {
        let fn = window.player.getCurrentFrame();

        var clsId = parseInt(clsBox.getAttribute("clsId"));
        clsId = clsId + 1;
        if (clsId > totalClasses - 1) {
            clsId = 0;
        }
        window.yByFrame[fn] = [fn, [0, 0]];
        window.yByFrame[fn][1][clsId] = 1.0;

        clsBox.setAttribute("clsId", clsId);
        clsBox.setAttribute("acc", "1.0");
        clsBox.setAttribute("fn", fn);
        updateLabel(clsBox, labels);
        drawChart(yByFrame);
    });


    prevClsId = -1;
    player.addEventListener("timeupdate", (t) => {
        let fn_y = window.yByFrame[t.frame];
        let y = fn_y[1];
        let fn = fn_y[0];
        if (fn == t.frame) {
            let clsId = argMax(y);
            let acc = y[clsId];
            // console.log(clsId, acc);

            if ((acc < 0.9 || prevClsId !== clsId) && player.isPlaying()) {
                console.log("force pause");
                player.pause();
            }

            clsBox.setAttribute("clsId", clsId);
            clsBox.setAttribute("acc", acc);
            clsBox.setAttribute("fn", fn);
            updateLabel(clsBox, labels);
        } else {
            console.warn("frames does not match", t, fn_y);
            window.player.seekFrame(fn);
        }
    });

});
