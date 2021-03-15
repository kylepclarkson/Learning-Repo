import React from 'react'
import {Button} from './Button'

import './Hero.css'

function Hero() {
    return (
        <div className='hero-container'>
            <video src="../videos/video-2.mp4" autoPlay loop muted/>
            <h1>Adventure awaits!</h1>
            <p>When will you start?</p> 
            <div className="hero-btns">
                <Button className='btns' buttonStyle='btn--outline' buttonSize='btn--large'>
                    Get Started
                </Button>
                <Button className='btns' buttonStyle='btn--primary' buttonSize='btn--large'>
                    Watch Trailer <i className='far fa-play-circle'/>
                </Button>
            </div>     
        </div>
    );
}

export default Hero;
