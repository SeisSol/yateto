/**
 * @file
 * This file is part of SeisSol.
 *
 * @author Alexander Heinecke (Alexander.Heinecke@mytum.de)
 * @author Sebastian Rettenberger (sebastian.rettenberger AT tum.de, http://www5.in.tum.de/wiki/index.php/Sebastian_Rettenberger)
 * @author Carsten Uphoff (c.uphoff AT tum.de, http://www5.in.tum.de/wiki/index.php/Carsten_Uphoff,_M.Sc.)
 *
 * @section LICENSE
 * Copyright (c) 2016-2017, SeisSol Group
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * @section DESCRIPTION
 * Stopwatch originally developed by A. Heinecke
 */

#ifndef STOPWATCH_H
#define STOPWATCH_H

#include <time.h>

/**
 * Stopwatch
 *
 * Part of SeisSol, so you can easily calculate the needed time of SeisSol computations with a high precision
 */
class Stopwatch
{
private:
	struct timespec m_start;

	/** Time already spent */
	long long m_time;
  
  /** Returns the time difference in nanoseconds. */
  long long difftime(struct timespec const& end)
  {
    return 1000000000L * (end.tv_sec - m_start.tv_sec) + end.tv_nsec - m_start.tv_nsec;
  }
  
  double seconds(long long time) 
  {
    return 1.0e-9 * time;
  }

public:
	/**
	 * Constructor
	 *
	 * resets the Stopwatch
	 */
	Stopwatch() : m_time(0)
  {}

	/**
	 * Destructor
	 */
	~Stopwatch()
	{}

	/**
	 * Reset the stopwatch to zero
	 */
	void reset()
	{
		m_time = 0;
	}

	/**
	 * starts the time measuring
	 */
	void start()
	{
		clock_gettime(CLOCK_MONOTONIC, &m_start);
	}

	/**
	 * get time measuring
	 *
	 * @return measured time (until now) in seconds
	 */
	double split()
	{
		struct timespec end;
		clock_gettime(CLOCK_MONOTONIC, &end);
    
    return seconds(difftime(end));
	}

	/**
	 * pauses the measuring
	 *
	 * @return measured time (until now) in seconds
	 */
	double pause()
	{
		struct timespec end;
		clock_gettime(CLOCK_MONOTONIC, &end);

		m_time += difftime(end);
		return seconds(m_time);
	}

	/**
	 * stops time measuring
	 *
	 * @return measured time in seconds
	 */
	double stop()
	{
		double time = pause();
		reset();
		return time;
	}
};

#endif // STOPWATCH_H

