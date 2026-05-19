(function () {
  var form = document.getElementById("predict-form");
  var out = document.getElementById("predict-result");
  var download = document.getElementById("download-csv");

  function showError(status, body) {
    if (!out) return;
    var msg = "[" + status + "] ";
    try {
      var parsed = JSON.parse(body);
      msg += parsed.error || body;
    } catch (_) {
      msg += body;
    }
    out.innerHTML =
      '<pre class="predict-error">' +
      msg.replace(/&/g, "&amp;").replace(/</g, "&lt;") +
      "</pre>";
  }

  if (form && out) {
    form.addEventListener("submit", function (e) {
      e.preventDefault();
      var file = document.getElementById("predict-file").files[0];
      if (!file) return;

      window.scrollTo(0, 0);
      out.textContent = "Running…";

      file
        .text()
        .then(function (csv) {
          return fetch("/predict", {
            method: "POST",
            headers: {
              "Content-Type": "text/csv",
              Accept: "text/html",
            },
            body: csv,
          });
        })
        .then(function (res) {
          var loc = res.headers.get("Content-Location");
          if (res.ok && loc) {
            // Real navigation — browser scrolls to top, the prediction page
            // is shareable, refreshable, and lives at a canonical URL.
            window.location.href = loc;
            return;
          }
          return res.text().then(function (body) {
            showError(res.status, body);
          });
        })
        .catch(function (err) {
          out.textContent = "Network error: " + err.message;
        });
    });
  }

  if (download) {
    var table = document.querySelector("table.predictions");
    if (!table) return;

    function csvCell(text) {
      var s = String(text).trim();
      if (/[",\n]/.test(s)) {
        return '"' + s.replace(/"/g, '""') + '"';
      }
      return s;
    }

    function tableToCsv(t) {
      var rows = [];
      var headerCells = t.querySelectorAll("thead tr th");
      if (headerCells.length) {
        var header = [];
        headerCells.forEach(function (c) {
          header.push(csvCell(c.textContent));
        });
        rows.push(header.join(","));
      }
      t.querySelectorAll("tbody tr").forEach(function (tr) {
        var cells = [];
        tr.querySelectorAll("td").forEach(function (td) {
          cells.push(csvCell(td.textContent));
        });
        rows.push(cells.join(","));
      });
      return rows.join("\n") + "\n";
    }

    download.addEventListener("click", function () {
      var csv = tableToCsv(table);
      var blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
      var url = URL.createObjectURL(blob);
      var a = document.createElement("a");
      a.href = url;
      a.download = "predictions.csv";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    });
  }
})();
