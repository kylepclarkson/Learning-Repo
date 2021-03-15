import React, {useState} from 'react'
import {Link} from 'react-router-dom'
import { Button } from './Button';

function Navbar() {

    // boolean. True if menu is open, false if
    const [click, setClick] = useState(false);
    // boolean. Displays button dependent of screen size
    const [button, setButton] = useState(true);

    // Toggle click
    const handleClick = () => setClick(!click);
    // close menu
    const closeMobileMenu = () => setClick(false);

    const showButton = () => {
        if(window.innerWidth <= 960) {
            setButton(false);
        } else {
            setButton(true);
        }
    };

    window.addEventListener('resize', showButton)

    return (
        <>
            <nav className="navbar">
                <div className="navbar-container">
                    <Link
                        to="/"
                        className="navbar-logo">
                            ExPLORE <i className="fab fa-typo3"/>
                    </Link>
                    <div className="menu-icon" onClick={handleClick}>
                        <i className={click ? 'fas fa-times' : 'fas fa-bars'}/>
                    </div>
                </div>
                <ul className={click ? 'nav-menu active': 'nav-menu'}>
                    <li className="nav-item">
                        <Link
                            to='/'
                            className='nav-links'
                            onClick={closeMobileMenu}>
                                Home
                        </Link>
                    </li>
                    <li className="nav-item">
                        <Link
                            to='/services'
                            className='nav-links'
                            onClick={closeMobileMenu}>
                                Services
                        </Link>
                    </li>
                    <li className="nav-item">
                        <Link
                            to='/products'
                            className='nav-links'
                            onClick={closeMobileMenu}>
                                Products
                        </Link>
                    </li>
                    <li className="nav-item">
                        <Link
                            to='/sign-up'
                            className='nav-links-mobile'
                            onClick={closeMobileMenu}>
                                Sign up
                        </Link>
                    </li>
                </ul>
                {button && <Button buttonStyle='btn-outline'>
                        Sign up
                    </Button>} 
            </nav>
        </>
    )
}

export default Navbar
