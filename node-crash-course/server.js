// import module
const http = require('http');
// create server. Function is run everytime a request is made to the server.
const server = http.createServer((request, response) => {
    console.log('Request made.')
    console.log(request.method)
});

// port, host, function on start
server.listen(3000, 'localhost', () => {
    console.log('listening on port 3000')
});

