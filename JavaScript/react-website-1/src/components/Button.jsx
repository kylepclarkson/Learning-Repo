import React from 'react'
import './Button.css'
import { Link } from 'react-router-dom'

// css classes that will be applied to buttons
const STYLES = ['btn--primary', 'btn--outline']
const SIZES = ['btn--medium', 'btn--large']

export const Button = ({ 
    children, 
    type, 
    onClick, 
    buttonStyle, 
    buttonSize,
}) => {
    // set style to provided style or default if not present.
    const checkButtonStyle = STYLES.includes(buttonStyle) ? buttonStyle : STYLES[0];
    const checkButtonSize = SIZES.includes(buttonSize) ? buttonSize : SIZES[0];

    return (
        <Link to='/' className='btn-mobile'>
            <button
                className={`btn ${checkButtonStyle} ${checkButtonStyle}`}
                onClick={onClick}
                type={type}>
                {children}
            </button>
        </Link>
    )
}
