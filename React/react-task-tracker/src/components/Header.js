import PropTypes from 'prop-types'

const Header = ({title}) => {
    return (
        <header>
            <h1>{title}</h1>
            <button>Add</button>
        </header>
    )
}

Header.defailtProps = {
    title: 'Task Trackers',
}

Header.PropTypes = {
    title: PropTypes.string.isRequired,
}

export default Header
