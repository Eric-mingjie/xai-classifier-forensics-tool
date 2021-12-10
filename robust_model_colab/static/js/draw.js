$(function() {

    function ellipse(context, cx, cy, rx, ry){
	    context.save(); // save state
	    context.beginPath();
	    context.translate(cx-rx, cy-ry);
	    context.scale(rx, ry);
	    context.arc(1, 1, 1, 0, 2 * Math.PI, false);
	    context.restore(); // restore to original state
	    context.stroke();
    }

    function initDraw(canvas) {

        var startX = 0;
        var startY = 0;
        var drawing = false;
        var dragDiv = document.getElementById("dragDiv");
        dragDiv.style.background = "#F00";
        dragDiv.style.position = "fixed";
        dragDiv.style.pointerEvents = "none";
        dragDiv.style.opacity = 0.5;

        canvas.onmouseup = function (e) {
            var rect = canvas.getBoundingClientRect();
            var x = e.clientX - rect.left;
            var y = e.clientY - rect.top;
        
            canvas.style.cursor = "default";
            console.log("finished.");
            var ctx = canvas.getContext("2d")
            ctx.fillStyle = color;
            var lenX = x - startX;
            var lenY = y - startY;
            ctx.beginPath();
            if (shape == 'rectangle') {
                ctx.rect(startX, startY, lenX, lenY);
            } else if (shape == 'circle') {
                ellipse(ctx, startX + lenX / 2, startY + lenY / 2, 
                            Math.abs(lenX) /2, Math.abs(lenY / 2));
                //ctx.arc(startX + lenX / 2, startY + lenY / 2,
                //        Math.abs(lenX) / 2, 0, 2 * Math.PI);
            }
            ctx.fill();
            drawing = false;
            dragDiv.style.display = "none";
            dragDiv.style.width = 0;
            dragDiv.style.height = 0;
            dragDiv.style.background = color;
        }

        canvas.onmousedown = function (e) {
            var rect = canvas.getBoundingClientRect();
            var x = e.clientX - rect.left;
            var y = e.clientY - rect.top;
        
            console.log("begun.");
            startX = x;
            startY = y;
            canvas.style.cursor = "crosshair";
            drawing = true;
            if (shape != "brush") {
                dragDiv.style.display = "block";
                dragDiv.style.top = e.clientY;
                dragDiv.style.left = e.clientX;
            }
            if (shape == "circle") {
                dragDiv.style.borderRadius = "50%";
            } else if (shape == "rectangle") {
                dragDiv.style.borderRadius = "0px";
            }
        }

        canvas.onmousemove = function (e) {
            if (!drawing) {
                return;
            }

            var rect = canvas.getBoundingClientRect();
            var x = e.clientX - rect.left;
            var y = e.clientY - rect.top;
            if (shape == "brush") {
                var ctx = canvas.getContext("2d");
                ctx.fillStyle = color;
                console.log(x, y);
                console.log(shape, drawing);
                ctx.beginPath();
                ctx.arc(x, y, 10, 0, 2 * Math.PI);
                ctx.fill();
            } 
            else {
                if (x < startX) {
                    dragDiv.style.right = rect.right - startX;
                    dragDiv.style.left = "";
                } else {
                    dragDiv.style.left = rect.left + startX;
                    dragDiv.style.right = "";
                }

                if (y < startY) {
                    dragDiv.style.bottom = "";
                    dragDiv.style.top = rect.top + y;
                } else {
                    dragDiv.style.top = rect.top + startY;
                    dragDiv.style.bottom = "";
                }

                dragDiv.style.width = Math.abs(x - startX);
                dragDiv.style.height = Math.abs(y - startY);
            }
        }
    }

    var color = "#000000";
    var shape = "rectangle";

    $("#myCanvas").ready(function(e) {
        // initDraw(document.getElementById('myCanvas'));
    });

    $("#colorpicker").ready(function(e) {
        color = "#" + $("#colorpicker").val();
    });
});
