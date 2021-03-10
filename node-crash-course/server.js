// import module
const http = require('http');
const fs = require('fs')

// create server. Function is run everytime a request is made to the server.
const server = http.createServer((request, response) => {
    // response.setHeader('Content-Type', 'text/plain') // send plain text
    response.setHeader('Content-Type', 'text/html')

    let path = './views';
    switch(request.url) {
        case '/':
            path += '/index.html'
            response.statusCode = 200;
            break;
        case '/about':
            path += '/about.html'
            response.statusCode = 200;
            break;
        case '/about-me':
            response.statusCode = 301
            // Redirect to /about
            response.setHeader('Location', '/about')
            response.end()
            break;
        default:
            path += '/404.html'
            response.statusCode = 404;
            break;
    }

    // send html file as response
    fs.readFile(path, (err, data) => {
        if (err) {
            console.log(err);
            response.end();
        } else {
            response.write(data);
            response.end()
        }
    })


});

// port, host, function on start
server.listen(3000, 'localhost', () => {
    console.log('listening on port 3000')
});

