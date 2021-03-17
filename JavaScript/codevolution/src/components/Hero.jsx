import React from 'react'

function Hero({heroName}) {
    if(heroName === 'Joker')
        throw new Error('Loser');
    return (
        <div>
            {heroName}
        </div>
    )
}

export default Hero
