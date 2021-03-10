const express = require('express');

// instace of express app.
const app = express();

// listen for requests.
app.listen(3000);

app.get('/', (req, res) => {
    // auto-infers content type (sets header) and status code.
    // res.send('<p>Home Page</p>')
    // sendFile expects absolute path; need to set root to this directory.
    res.sendFile('./views/index.html', { root: __dirname})
});

app.get('/about', (req, res) => {
    // auto-infers content type (sets header) and status code.
    // res.send('<p>About Page</p>')
    res.sendFile('./views/about.html', { root: __dirname})
});

// redirect
app.get('/about-us', (req, res) => {
    res.redirect('/about');
})

// 404
// use fires for all requests, only if the functions above have not been fired.
app.use((req, res) => {
    res.status(404).sendFile('./views/404.html', { root: __dirname})
})
