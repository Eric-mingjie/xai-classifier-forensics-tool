$(function() {
    // var slider0 = document.getElementById("clf-id");
    // var output0 = document.getElementById("demo-clf-id");
    // output0.innerHTML = slider0.value; // Display the default slider value
    
    // // Update the current slider value (each time you drag the slider handle)
    // slider0.oninput = function() {
    //   output0.innerHTML = this.value;
    // }

    var slider = document.getElementById("epsilon");
    var output = document.getElementById("demo");
    output.innerHTML = slider.value; // Display the default slider value
    
    // Update the current slider value (each time you drag the slider handle)
    slider.oninput = function() {
      output.innerHTML = this.value;
    }

    var clf_selector = document.getElementById("clfselector")

    // var slider2 = document.getElementById("steps");
    // var output2 = document.getElementById("demo2");
    // output2.innerHTML = slider2.value; // Display the default slider value
    
    // // Update the current slider value (each time you drag the slider handle)
    // slider2.oninput = function() {
    //   output2.innerHTML = this.value;
    // }

    // var num_noise_vec = document.getElementById("noiseVec");

    // var tv = document.getElementById("totalVariation");

    var patch_selector = document.getElementById("patchselector");
    // console.log(patch_selector.value);
    // ///////////////////////////////////////////////////////////////////
    var imagenet_label_dict = {
        "0": "whale",
        "1": "whippet",
        "2": "hyena",
        "3": "coucal",
        "4": "sax", 
    }

    clf_selector.onchange = function(e){
        if (clf_selector.value.includes("ImageNet")) {
            for (var i=0; i<5; i++){
                $(".cleanclass").get(i).innerHTML = imagenet_label_dict[i.toString()];
                $(".adversarialclass").get(i).innerHTML = imagenet_label_dict[i.toString()];
            }
        }
        else {
            for (var i=0; i<5; i++){
                $(".adversarialclass").get(i).innerHTML = "Label "+i;
                $(".cleanclass").get(i).innerHTML = "Label "+i;
            }
        }
    };

    var startX_log=0;
    var startY_log=0;
    var lenX_log=0;
    var lenY_log=0;

    function initDraw_v2(canvas_v2) {
        console.log("initDraw_v2 start");
        var startX = 0;
        var startY = 0;
        var drawing = false;

        canvas_v2.onmouseup = function (e) {
            console.log("test start");
            var rect = canvas_v2.getBoundingClientRect();
            var x = e.clientX - rect.left;
            var y = e.clientY - rect.top;

            canvas_v2.style.cursor = "default";
            console.log("finished.");
            var ctx = canvas_v2.getContext("2d");
            var lenX = x - startX;
            var lenY = y - startY;
            ctx.beginPath();
            ctx.strokeRect(startX, startY, lenX, lenY);

            ctx.stroke();

            drawing = false;
            
            rect_log_v2.lenX = lenX;
            rect_log_v2.lenY = lenY;

            cropped_patch_source_canvas_id = current_canvas_id;

            console.log("ffffffffffffffff");
            console.log(rect_log_v2.startX);
            console.log(rect_log_v2.startY);
            console.log(rect_log_v2.lenX);
            console.log(rect_log_v2.lenY);
            console.log("ggggggggggggggggg");

            // lenX_log = lenX;
            // lenY_log = lenY;
        }

        canvas_v2.onmousedown = function (e) {
            var rect = canvas_v2.getBoundingClientRect();
            var x = e.clientX - rect.left;
            var y = e.clientY - rect.top;

            console.log("begun.");
            startX = x;
            startY = y;
            canvas_v2.style.cursor = "crosshair";
            drawing = true;

            rect_log_v2.startX = x;
            rect_log_v2.startY = y;

            // startX_log = startX;
            // startY_log = startY;
        }

        canvas_v2.onmousemove = function (e) {
            if (!drawing) {
                return;
            }

            var rect = canvas_v2.getBoundingClientRect();
            var x = e.clientX - rect.left;
            var y = e.clientY - rect.top;
        }
    }

    var rect_log = new Proxy({}, {
        get: (target, name) => name in target ? target[name] : 0
      });

    var rect_log_v2 = new Proxy({}, {
        get: (target, name) => name in target ? target[name] : 0
      });


    // /////////////////////////////////////////////////////////////////////////////

    // document.getElementById("loading").style.display = "none";

    function handleClick(e) {
        console.log("Handling Click");
        // encoded_image = document.getElementById("myCanvas").toDataURL();
        // document.getElementById("loading").style.display = "block";

        $.post("http://localhost:{{port}}/toclass", JSON.stringify({
            // "id": slider0.value,
            "clf": clf_selector.value,
            "targeted": document.getElementById("ifTargeted").checked,
            "target": document.getElementById("targetClass").value,
            "epsilon": slider.value,
            // "steps": slider2.value,
            // "num_noise_vec": num_noise_vec.value,
            // "tv": tv.value,
            "startX": rect_log.startX/500.,
            "startY": rect_log.startY/500.,
            "lenX": rect_log.lenX/500.,
            "lenY": rect_log.lenY/500.
            }), function(data, stat) {
                    console.log("return success");
                    // var res = data['smoothed_clean_pred_class'];
                    // // var res2 = data['orig_clean_pred_class'];
                    // for (var i=0; i<25; i++){
                    //     if (clf_selector.value.includes("TrojAI")){
                    //         $(".clean_prediction")[i].innerHTML="Label "+res[i];
                    //     }
                    //     else {
                    //         $(".clean_prediction")[i].innerHTML=imagenet_label_dict[res[i]]; 
                    //     }
                    //      if (parseInt(res[i]) == parseInt(i/5)){
                    //         $(".clean_prediction")[i].style.color = "green";
                    //     }
                    //     else {
                    //         $(".clean_prediction")[i].style.color = "red";
                    //     }
                    // }

                    console.log("return sucess 2");

                    res = data['smoothed_adv_pred_class'];
                    // res2 = data['orig_adv_pred_class'];
                    for (var i=0; i<25; i++){
                        if (clf_selector.value.includes("TrojAI")){
                            $(".adv_prediction")[i].innerHTML="Label "+res[i];
                        }
                        else {
                            $(".adv_prediction")[i].innerHTML=imagenet_label_dict[res[i]]; 
                        }
                        if (parseInt(res[i]) == parseInt(i/5)){
                            $(".adv_prediction")[i].style.color = "green";
                        }
                        else {
                            $(".adv_prediction")[i].style.color = "red";
                        }
                    }

                    console.log("returne success 3");

                    img_list = data['adv_image_list'];

                    var images = [];
                    var img;
                    for (var i=0; i<img_list.length; i++){
                        img = new Image();
                        img.src = 'data:image/png;base64,' + img_list[i];
                        img.style.imageRendering = 'pixelated';
                        images.push(img);
                    }

                    Promise.all(images.map(function(image) {
                        return new Promise(function(resolve, reject) {
                          image.onload = resolve;
                        });
                      }))
                      .then(function() {
                        for (var i = 0; i < images.length; i++) {
                            x = Math.floor(i/5);
                            y = i % 5;
                            var img = images[i];
                            $('.advCanvas').get(i).getContext("2d").drawImage(img, 0, 0, 100, 100);
                        }
                        // document.getElementById("loading").style.display = "none";
                    });
                }
            );
    }

    var timeoutId = 0;

    $(".advCanvas").ready(function(e) {
        var clean_canvas = $(".advCanvas");
        var ctx;
        for (var i=0; i<clean_canvas.length; i++) {
            ctx = clean_canvas.get(i).getContext("2d");
            ctx.fillStyle = "#CCCCCC";
            ctx.fillRect(0, 0, 100, 100);
        };
    })

    /* ############################################################################## */
    var color = "#"+$("#colorpicker").val();

    $("#baseCanvas").ready(function(e) {
        var base_canvas = $("#baseCanvas");
        var ctx;
        for (var i=0; i<base_canvas.length; i++) {
            ctx = base_canvas.get(i).getContext("2d");
            ctx.fillStyle = "#CCCCCC";
            ctx.fillRect(0, 0, 450, 450);
        };

        $(".sampleImage").on('click', function(e) {
            console.log("start");
            var img = document.createElement("img");
            img.src = "http://localhost:{{port}}/static/pics/"+clf_selector.value+"/class_0_example_0";
            img.onload = function(){
                $('#baseCanvas')[0].getContext("2d").drawImage(img, 0, 0, 450, 450); 
            };

            var images = [];
            for (var i=0; i<5; i++){
                for (var j=0;j<5; j++){
                    images.push("http://localhost:{{port}}/static/pics/"+clf_selector.value+"/class_"+i+"_example_"+j);
                }
            }

            images = images.map(function(i) {
                var img = document.createElement("img");
                img.src = i;
                return img;
            });

            Promise.all(images.map(function(image) {
                return new Promise(function(resolve, reject) {
                  image.onload = resolve;
                });
              }))
              .then(function() {
                for (var i = 0; i < images.length; i++) {
                    x = Math.floor(i/5);
                    y = i % 5;
                    var img = images[i];
                    $('.poisonCanvas')[i].getContext("2d").drawImage(img, 0, 0, 95, 95);
                }
            });

            console.log("clean images loaded");

            for (var i=0; i<25; i++) {
                $(".poison_prediction")[i].innerHTML="";
            }

            lenX_log = 0;
            lenY_log = 0;
            startX_log = 0;
            startY_log = 0;
        })

        $(".apply-button").on('click', function(e) {
            if (patch_selector.value == "color patch"){
                var poisonCanvas = $(".poisonCanvas");
                var ratio = poisonCanvas[0].width / $("#baseCanvas")[0].width;

                for (var i=0; i < poisonCanvas.length; i++){
                    var ctx = poisonCanvas.get(i).getContext("2d");
                    ctx.fillStyle = color;
                    ctx.beginPath();
                    ctx.fillRect(startX_log * ratio, startY_log * ratio, lenX_log * ratio, lenY_log * ratio);
                    ctx.fill();
                }

                for (var i=0; i<25; i++){
                    $(".poison_prediction")[i].innerHTML="";//+" / "+res2[i];
                }
            } else {
                var ratio = 224. / 500.;
                // var ratio2 = $(".poisonCanvas")[0].width / $("#baseCanvas")[0].width;
                var ratio2 = 224./450.;
                console.log(rect_log_v2.startX);
                console.log(rect_log_v2.startY);
                console.log(rect_log_v2.lenX);
                console.log(rect_log_v2.lenY);
                $.post("http://localhost:{{port}}/crop", JSON.stringify({
                    // "clf_id": slider0.value,
                    "clf": clf_selector.value, 
                    "startX": rect_log_v2.startX*ratio,
                    "startY": rect_log_v2.startY*ratio,
                    "lenX": rect_log_v2.lenX*ratio,
                    "lenY": rect_log_v2.lenY*ratio,
                    "startX_v2": startX_log * ratio2,
                    "startY_v2": startY_log * ratio2,
                    "canvas_id_str": cropped_patch_source_canvas_id,
                    }), function(data, stats) {
                        img_list = data['out_img_list'];

                        var images = [];
                        var img;
                        for (var i=0; i<img_list.length; i++){
                            img = new Image();
                            img.src = 'data:image/png;base64,' + img_list[i];
                            img.style.imageRendering = 'pixelated';
                            images.push(img);
                        }
    
                        Promise.all(images.map(function(image) {
                            return new Promise(function(resolve, reject) {
                              image.onload = resolve;
                            });
                          }))
                          .then(function() {
                            for (var i = 0; i < images.length; i++) {
                                x = Math.floor(i/5);
                                y = i % 5;
                                var img = images[i];
                                $('.poisonCanvas').get(i).getContext("2d").drawImage(img, 0, 0, 95, 95);
                            }
                        });
                    }
                );
            }
        })

        $(".predict-button").on('click', function(e) {
            var ratio1 = 224. / $("#baseCanvas")[0].width;
            var ratio2 = 224. / 500.;
            var startX, startY, lenX, lenY;
            if (patch_selector.value == "color patch"){
                startX = startX_log*ratio1;
                startY = startY_log*ratio1;
                lenX = lenX_log*ratio1;
                lenY = lenY_log*ratio1;

                startX_v2 = startX;
                startY_v2 = startY;
            } else {
                startX = rect_log_v2.startX*ratio2;
                startY = rect_log_v2.startY*ratio2;
                lenX = rect_log_v2.lenX*ratio2;
                lenY = rect_log_v2.lenY*ratio2;
                
                startX_v2 = startX_log * ratio1;
                startY_v2 = startY_log * ratio1;
            }
            $.post("http://localhost:{{port}}/predict", JSON.stringify({
                // "clf_id":  slider0.value,
                "clf": clf_selector.value,
                "startX": startX,
                "startY": startY,
                "lenX": lenX,
                "lenY": lenY,
                "startX_v2": startX_v2,
                "startY_v2": startY_v2,
                "color": color,
                "patch_option": patch_selector.value,
                "canvas_id_str": cropped_patch_source_canvas_id,
                }), function(data, stats) {
                    var pred = data["prediction"];
                    for (var i=0; i<25; i++){
                        if (clf_selector.value.includes("TrojAI")){
                            $(".poison_prediction")[i].innerHTML="Label "+pred[i];
                        }
                        else {
                            $(".poison_prediction")[i].innerHTML=imagenet_label_dict[pred[i]]; 
                        }

                        if (parseInt(pred[i]) == parseInt(i/5)){
                            $(".poison_prediction")[i].style.color = "green";
                        }
                        else {
                            $(".poison_prediction")[i].style.color = "red";
                        }
                    }
                }
            );
        })

        $("#colorpicker").on('change', function(e) {
            color = "#" + e.target.value;

            for (var i=0; i<25; i++){
                $(".poison_prediction")[i].innerHTML="";//+" / "+res2[i];
            }

            if (lenX_log != 0 && lenY_log !=0){
                ctx = $("#baseCanvas")[0].getContext("2d");
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.fillRect(startX_log, startY_log, lenX_log, lenY_log);
    
                ctx.fill();
            }
        });

        var canvas = base_canvas[0];
        canvas.onmousedown = function (e) {
            var startX = 0;
            var startY = 0;
            var drawing = false;

            var rect = canvas.getBoundingClientRect();
            var x = e.clientX - rect.left;
            var y = e.clientY - rect.top;
        
            console.log("begun.");
            startX = x;
            startY = y;
            canvas.style.cursor = "crosshair";

            startX_log = x;
            startY_log = y; 
            if (patch_selector.value == "color patch"){
                canvas.onmouseup = function (e) {
                    var rect = canvas.getBoundingClientRect();
                    var x = e.clientX - rect.left;
                    var y = e.clientY - rect.top;
        
                    canvas.style.cursor = "default";
                    console.log("finished.");
                    var ctx = canvas.getContext("2d");
                    ctx.fillStyle = color;
                    var lenX = x - startX;
                    var lenY = y - startY;
                    ctx.beginPath();
                    ctx.fillRect(startX, startY, lenX, lenY);
        
                    ctx.fill();
                    drawing = false;
                    
                    rect_log.lenX = lenX;
                    rect_log.lenY = lenY;
        
                    lenX_log = lenX;
                    lenY_log = lenY;
                } 

                canvas.onmousemove = function (e) {
                    if (!drawing) {
                        return;
                    }
        
                    var rect = canvas.getBoundingClientRect();
                    var x = e.clientX - rect.left;
                    var y = e.clientY - rect.top;
                }
            }
            else if (patch_selector.value == "cropped patch"){
                // canvas.onmouseup = function(e) {
                    canvas.onmouseup = function(e){
                    };
                    canvas.onmousemove = function(e){
                    };

                    var ratio = 224. / 500;
                    var ratio2 = 224. / 450.;

                    console.log("zed");
                    console.log(startX_log);
                    console.log(startY_log);
                    console.log("diana");
                    $.post("http://localhost:{{port}}/crop2", JSON.stringify({
                        // "clf_id": slider0.value,
                        "clf": clf_selector.value, 
                        "startX": rect_log_v2.startX*ratio,
                        "startY": rect_log_v2.startY*ratio,
                        "lenX": rect_log_v2.lenX*ratio,
                        "lenY": rect_log_v2.lenY*ratio,
                        "startX_v2": startX_log * ratio2,
                        "startY_v2": startY_log * ratio2,
                        "canvas_id_str": cropped_patch_source_canvas_id,
                        }), function(data, stats) {
                            console.log("return");
                            img_list = data['out_img_list'];
    
                            var images = [];
                            var img;
                            for (var i=0; i<img_list.length; i++){
                                img = new Image();
                                img.src = 'data:image/png;base64,' + img_list[i];
                                img.style.imageRendering = 'pixelated';
                                images.push(img);
                            }
                        
                            $('#baseCanvas')[0].getContext("2d").drawImage(images[0], 0, 0, 450, 450);                    
                        }
                    ); 
                // };
            }
        }

    })

    patch_selector.onchange = function(e) {
        if (patch_selector.value == "cropped patch"){
            document.getElementById("control-color-selector").style.display = "none";
        }
        else{
            document.getElementById("control-color-selector").style.display = "block"; 
        }
    }

    $(".poisonCanvas").ready(function(e) {
        var clean_canvas = $(".poisonCanvas");
        var ctx;
        for (var i=0; i<clean_canvas.length; i++) {
            ctx = clean_canvas.get(i).getContext("2d");
            ctx.fillStyle = "#CCCCCC";
            ctx.fillRect(0, 0, 95, 95);
        };
    })

    /* ############################################################################## */

    $(".cleanCanvas").ready(function(e) {
        var clean_canvas = $(".cleanCanvas");
        var ctx;
        for (var i=0; i<clean_canvas.length; i++) {
            ctx = clean_canvas.get(i).getContext("2d");
            ctx.fillStyle = "#CCCCCC";
            ctx.fillRect(0, 0, 95, 95);
        };

        // clean_canvas.get(0).getContext("2d").fillText('Hello world', 100, 50);

        $(".resetanchor").on('click', function(e) {
            var adv_canvas = $(".advCanvas");
            var ctx;
            for (var i=0; i<adv_canvas.length; i++) {
                ctx = adv_canvas.get(i).getContext("2d");
                ctx.fillStyle = "#CCCCCC";
                ctx.fillRect(0, 0, 95, 95);
            };

            for (var i=0; i<25; i++){
                $(".adv_prediction")[i].innerHTML="";
            };

            rect_log.startX = 0;
            rect_log.startY = 0;
            rect_log.lenX = 0;
            rect_log.lenY = 0;
        });

        $(".imgbutton").on('click', function(e) {
            var adv_canvas = $(".advCanvas");
            var ctx;
            for (var i=0; i<adv_canvas.length; i++) {
                ctx = adv_canvas.get(i).getContext("2d");
                ctx.fillStyle = "#CCCCCC";
                ctx.fillRect(0, 0, 95, 95);
            };

            var poison_canvas = $(".poisonCanvas");
            for (var i=0; i<poison_canvas.length; i++) {
                ctx = poison_canvas.get(i).getContext("2d");
                ctx.fillStyle = "#CCCCCC";
                ctx.fillRect(0, 0, 95, 95);
            };

            var base_canvas = $("#baseCanvas");
            for (var i=0; i<base_canvas.length; i++) {
                ctx = base_canvas.get(i).getContext("2d");
                ctx.fillStyle = "#CCCCCC";
                ctx.fillRect(0, 0, 450, 450);
            };

            for (var i=0; i<25; i++){
                $(".clean_prediction")[i].innerHTML="";
                $(".adv_prediction")[i].innerHTML="";
            }

            var images = [];
            for (var i=0; i<5; i++){
                for (var j=0;j<5; j++){
                    images.push("http://localhost:{{port}}/static/pics/"+clf_selector.value+"/class_"+i+"_example_"+j);
                }
            }

            images = images.map(function(i) {
                var img = document.createElement("img");
                img.src = i;
                return img;
            });

            Promise.all(images.map(function(image) {
                return new Promise(function(resolve, reject) {
                  image.onload = resolve;
                });
              }))
              .then(function() {
                for (var i = 0; i < images.length; i++) {
                    x = Math.floor(i/5);
                    y = i % 5;
                    var img = images[i];
                    $('.cleanCanvas')[i].getContext("2d").drawImage(img, 0, 0, 95, 95);
                }
            });
    
            $.post("http://localhost:{{port}}/cleanpredict", JSON.stringify({
                "clf": clf_selector.value,
                }), function(data, stat) {
                    console.log("return success");
                    var res = data['clean_pred_class'];
                    // var res2 = data['orig_clean_pred_class'];
                    for (var i=0; i<25; i++){
                        if (clf_selector.value.includes("TrojAI")){
                            $(".clean_prediction")[i].innerHTML="Label "+res[i];
                        }
                        else {
                            $(".clean_prediction")[i].innerHTML=imagenet_label_dict[res[i]]; 
                        }
                         if (parseInt(res[i]) == parseInt(i/5)){
                            $(".clean_prediction")[i].style.color = "green";
                        }
                        else {
                            $(".clean_prediction")[i].style.color = "red";
                        }
                    }
                });

            // initDraw(document.getElementById('img01'));

            rect_log.startX = 0;
            rect_log.startY = 0;
            rect_log.lenX = 0;
            rect_log.lenY = 0;

            rect_log_v2.startX = 0;
            rect_log_v2.startY = 0;
            rect_log_v2.lenX = 0;
            rect_log_v2.lenY = 0;
        });

        $(".setanchor").on('mousedown', function(e) {
            timeoutId = setInterval(function(){ handleClick(e); }, 250);
        }).on('mouseup mouseleave', function() {
            clearTimeout(timeoutId);
        }).on('click', function(e) {
            handleClick(e);
        });
    });

    // ///////////////////////////////////////////////////////////////////////////////////
    var modal = document.getElementById("myModal");

    // Get the image and insert it inside the modal - use its "alt" text as a caption
    var modalImgCanvas = document.getElementById("img01");
    var current_canvas_id = null;
    var cropped_patch_source_canvas_id = null;
    // img.onclick = function(){
    //   modal.style.display = "block";
    //   modalImg.src = img.toDataURL();
    // }

    $('#img01').ready(function(e) {
        var base_canvas = $("#img01");
        initDraw_v2(base_canvas[0]);
    });

    $(".cleanCanvas").on("click", function() {
        modal.style.display = "block";
        // modalImg.src = this.toDataURL(); 
        var img = new Image();
        img.src = this.toDataURL();
        img.onload = function() {
            var ctx = modalImgCanvas.getContext("2d");
            ctx.clearRect(0, 0, 500, 500);
            ctx.drawImage(img, 0, 0, 500, 500);

            if (rect_log.lenX != 0 && rect_log.lenY != 0) {
                ctx.beginPath();
                ctx.strokeRect(rect_log.startX, rect_log.startY, rect_log.lenX, rect_log.lenY);
            }
        };
    });

    $(".advCanvas").on("click", function() {
        // console.log("advCanvas clicked");
        // console.log(this.id);
        current_canvas_id = this.id;
        modal.style.display = "block";
        var img = new Image();
        img.src = this.toDataURL();
        img.onload = function() {
            var ctx = modalImgCanvas.getContext("2d");
            ctx.clearRect(0, 0, 500, 500);
            ctx.drawImage(img, 0, 0, 500, 500);
        };
    });

    // Get the <span> element that closes the modal
    var span = document.getElementsByClassName("close")[0];
    
    // When the user clicks on <span> (x), close the modal
    span.onclick = function() { 
      modal.style.display = "none";
    }

    ///////////////////////////////////////////////////////////////////////////
    $('#img01').mouseup(function(e) {
        var rect = this.getBoundingClientRect();
        var x = e.clientX - rect.left;
        var y = e.clientY - rect.top;

        var c = this.getContext('2d');
        var coord = "x=" + x + ", y=" + y;
        var p = c.getImageData(x, y, 1, 1).data;
        var hex = "#" + ("000000" + rgbToHex(p[0], p[1], p[2])).slice(-6);

        $('#color-status').html(coord + "<br>" + hex);
    });

    function findPos(obj) {
        var curleft = 0, curtop = 0;
        if (obj.offsetParent) {
            do {
                curleft += obj.offsetLeft;
                curtop += obj.offsetTop;
            } while (obj = obj.offsetParent);
            return { x: curleft, y: curtop };
        }
        return undefined;
    }
    
    function rgbToHex(r, g, b) {
        if (r > 255 || g > 255 || b > 255)
            throw "Invalid color component";
        return ((r << 16) | (g << 8) | b).toString(16);
    }
    
    function randomInt(max) {
      return Math.floor(Math.random() * max);
    }
    
    function randomColor() {
        return `rgb(${randomInt(256)}, ${randomInt(256)}, ${randomInt(256)})`
    }
});