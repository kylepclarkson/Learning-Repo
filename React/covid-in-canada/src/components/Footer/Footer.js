import React from 'react'

import { Typography } from '@material-ui/core'

import './Footer.css'

const Footer = () => {
  return (
    <div className='main-footer'>
      <div className="container">
        <div>
          <Typography color='white' variant='h4' className={'row'}>
            Created by <a href={'https://kyleclarkson.ca'} target='_blank' className='text-white'>Kyle Clarkson</a>.
          </Typography>
          <Typography color='white' variant='body2'>
            *Using 2021 population estimates.
          </Typography>
          <Typography color='white' variant='body2'>
            **An average of the previous seven day values.
          </Typography>
          <Typography color='white' variant='body1' className={'row'}>
            Thanks to the team at <a href={'https://opencovid.ca/api/'} target='_blank' className={'text-white'}>open covid</a> which supplied the data!
          </Typography>
        </div>
      </div>
    </div>
  )
}

export default Footer
