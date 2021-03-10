const fs = require('fs')

const readStream = fs.createReadStream('./docs/blog3.txt', {encoding: 'utf-8'})
const writeStream = fs.createWriteStream('./docs/blog4.txt')

// get 'chunks' of data from the stream. Each time a chunk arrives, run function.
// readStream.on('data', (chunk) => {
//     console.log('=============== new chunk ==============')
//     console.log(chunk)
//     writeStream.write('\n ========== New Chunk ============== \n')
//     writeStream.write(chunk)
// })

/**
 * Piping read from read stream to write stream.
 */
readStream.pipe(writeStream);